package main

import (
	"fmt"
	"math/rand"
	"os"
	"os/exec"
	"slices"
	"strings"
)

var backend int

const (
	be_golang = iota + 1
	be_cuda
)

func initBackend() {
	switch {
	case os.Getenv("WHALE_GO") == "1":
		backend = be_golang
		return
	case os.Getenv("WHALE_CUDA") == "1":
		backend = be_cuda
		return
	}

	available := func(cmd string) bool {
		err := exec.Command("bash", "-c", "command -v "+cmd).Run()
		return err == nil // likely the cmd is available
	}

	if available("nvcc") {
		backend = be_cuda
		return
	}

	backend = be_golang
}

/*
 * Generate a sequence of IR from tensor AST
 */

func generateIR(t *Tensor, gpu bool) (irs []*instruction, err error) {
	defer func() {
		if r := recover(); r != nil {
			err = fmt.Errorf("%v", r.(string))
			irs = nil
		}
	}()

	inst := func(m mnemonic) *instruction {
		return &instruction{id: newInstid(), mnemonic: m}
	}

	kernels := []*instruction{}
	pushK := func(ir *instruction) instid {
		kernels = append(kernels, ir)
		return ir.id
	}

	entries := []*instruction{}
	pushE := func(ir *instruction) instid {
		entries = append(entries, ir)
		return ir.id
	}

	pushE(inst(&mnEntry{}))

	var dfs func(t *Tensor) instid
	dfs = func(t *Tensor) instid {
		switch t.op {
		case op_const:
			return pushE(inst(&mnParam{typ: t_floats, val: t.data}))

		case op_recip:
			input := t.inputs[0]
			size := input.Size()

			inputid := dfs(input)

			/*
			 * define kernel
			 */

			var kern instid

			if gpu {
				paramx := pushK(inst(&mnKernParam{typ: t_floats}))
				paramresult := pushK(inst(&mnKernParam{typ: t_floats}))
				kern = pushK(inst(&mnKernel{params: []instid{paramx, paramresult}}))
				idx := pushK(inst(&mnThreadPosition{dimensions: 1}))
				target := pushK(inst(&mnInit{from: paramx, idx: idx}))
				var op alu1op
				if t.op == op_recip {
					op = alu1_recip
				}
				alu1 := pushK(inst(&mnALU1{val: target, op: op}))
				pushK(inst(&mnAssign{left: paramresult, lidx: idx, right: alu1}))
				pushK(inst(&mnEndKernel{}))

			} else {
				paramIdx := pushK(inst(&mnKernParam{typ: t_int}))
				paramx := pushK(inst(&mnKernParam{typ: t_floats}))
				paramresult := pushK(inst(&mnKernParam{typ: t_floats}))
				kern = pushK(inst(&mnKernel{params: []instid{paramIdx, paramx, paramresult}}))
				target := pushK(inst(&mnInit{from: paramx, idx: paramIdx}))
				var op alu1op
				if t.op == op_recip {
					op = alu1_recip
				}
				alu1 := pushK(inst(&mnALU1{val: target, op: op}))
				pushK(inst(&mnAssign{left: paramresult, lidx: paramIdx, right: alu1}))
				pushK(inst(&mnEndKernel{}))
			}

			/*
			 * call kernel from entry
			 */

			// define result to store
			result := pushE(inst(&mnDecl{typ: t_floats, length: size}))

			if gpu {
				pushE(inst(&mnInvokeKernel{
					kernel:         kern,
					parallelLevel1: &kernelParallelizationParam{x: 1},
					parallelLevel2: &kernelParallelizationParam{x: size},
					args:           []instid{inputid, result},
				}))
			} else {
				// start loop and invokes kernel
				loop := pushE(inst(&mnLoop{countImm: size}))
				pushE(inst(&mnInvokeKernel{kernel: kern, args: []instid{loop, inputid, result}}))
			}

			pushE(inst(&mnEndLoop{}))

			return result

		case op_add, op_mul:
			l, r := t.inputs[0], t.inputs[1]
			sizel, sizer := l.Size(), r.Size()
			if sizel != sizer {
				panic(fmt.Sprintf("cannot compute 2 different size tensors: %v and %v", l, r))
			}

			// get left and right id first
			lid, rid := dfs(l), dfs(r)

			/*
			 * define kernel
			 */

			var kern instid

			if gpu {
				paraml := pushK(inst(&mnKernParam{typ: t_floats}))
				paramr := pushK(inst(&mnKernParam{typ: t_floats}))
				paramresult := pushK(inst(&mnKernParam{typ: t_floats}))
				kern = pushK(inst(&mnKernel{params: []instid{paraml, paramr, paramresult}}))

				// assume vector
				// todo: support 2 or more dimensions

				idx := pushK(inst(&mnThreadPosition{dimensions: 1}))

				// compute stride, considering broadcast
				lstride := pushK(inst(&mnInitImm{typ: t_int, val: l.dim.strides[0]}))
				rstride := pushK(inst(&mnInitImm{typ: t_int, val: r.dim.strides[0]}))

				// define index
				lidx := pushK(inst(&mnALU2{left: idx, op: alu2_mul, right: lstride}))
				ridx := pushK(inst(&mnALU2{left: idx, op: alu2_mul, right: rstride}))

				// load value to be computed from left and right
				loadl := pushK(inst(&mnInit{from: paraml, idx: lidx}))
				loadr := pushK(inst(&mnInit{from: paramr, idx: ridx}))

				var op alu2op
				if t.op == op_add {
					op = alu2_add
				} else {
					op = alu2_mul
				}

				// do compute
				alu2 := pushK(inst(&mnALU2{left: loadl, op: op, right: loadr}))

				// assign computed to result
				pushK(inst(&mnAssign{left: paramresult, lidx: idx, right: alu2}))

				// finish kernel
				pushK(inst(&mnEndKernel{}))
			} else {
				paramIdx := pushK(inst(&mnKernParam{typ: t_int}))
				paraml := pushK(inst(&mnKernParam{typ: t_floats}))
				paramr := pushK(inst(&mnKernParam{typ: t_floats}))
				paramresult := pushK(inst(&mnKernParam{typ: t_floats}))
				kern = pushK(inst(&mnKernel{params: []instid{paramIdx, paraml, paramr, paramresult}}))

				// assume vector
				// todo: support 2 or more dimensions

				// compute stride, considering broadcast
				lstride := pushK(inst(&mnInitImm{typ: t_int, val: l.dim.strides[0]}))
				rstride := pushK(inst(&mnInitImm{typ: t_int, val: r.dim.strides[0]}))

				// define index
				lidx := pushK(inst(&mnALU2{left: paramIdx, op: alu2_mul, right: lstride}))
				ridx := pushK(inst(&mnALU2{left: paramIdx, op: alu2_mul, right: rstride}))

				// load value to be computed from left and right
				loadl := pushK(inst(&mnInit{from: paraml, idx: lidx}))
				loadr := pushK(inst(&mnInit{from: paramr, idx: ridx}))

				var op alu2op
				if t.op == op_add {
					op = alu2_add
				} else {
					op = alu2_mul
				}

				// do compute
				alu2 := pushK(inst(&mnALU2{left: loadl, op: op, right: loadr}))

				// assign computed to result
				pushK(inst(&mnAssign{left: paramresult, lidx: paramIdx, right: alu2}))

				// finish kernel
				pushK(inst(&mnEndKernel{}))
			}

			/*
			 * call kernel from entry
			 */

			// define result to store
			result := pushE(inst(&mnDecl{typ: t_floats, length: sizel}))

			if gpu {
				pushE(inst(&mnInvokeKernel{
					kernel:         kern,
					parallelLevel1: &kernelParallelizationParam{x: 1},
					parallelLevel2: &kernelParallelizationParam{x: sizel},
					args:           []instid{lid, rid, result},
				}))
			} else {
				// start loop and invoke kernel
				loop := pushE(inst(&mnLoop{countImm: sizel}))
				pushE(inst(&mnInvokeKernel{kernel: kern, args: []instid{loop, lid, rid, result}}))
			}

			pushE(inst(&mnEndLoop{}))

			return result

		case op_expand:
			return dfs(t.inputs[0])

		default:
			panic(fmt.Sprintf("unknown op: %v", t.op))
		}
	}

	result := dfs(t)
	pushE(inst(&mnReturn{val: result}))
	pushE(inst(&mnEndEntry{}))

	return slices.Concat(kernels, entries), nil
}

/*
 * dll; in whale, generated ir is rendered by renderer as dll and executor executes it.
 */

type dllparam struct {
	name  string
	value []float32
}

type dll struct {
	src        string
	entrypoint string
	params     []*dllparam
}

/*
 * renderer
 */

type renderer interface {
	render(irs []*instruction) (*dll, error)
}

type idx struct {
	val string
}

type cLikeLangRenderer interface {
	indent() string
	header() string

	varname(id int) string

	kernelName(id int) string
	kernel(kernname string, params []string, typs []typ) string
	endKernel() string

	invokeKernel(kernname string, args []string) string

	entrypointName() string
	entrypoint(entryname string) string
	endEntrypoint() string

	// ir
	global(varname string, typ typ) string
	return_(varname string) string
	decl(varname string, typ typ, length int) string
	initImm(varname string, imm any, typ typ) string
	init_(varname string, from string, idx *idx) string
	assign(left, right string, lidx, ridx *idx) string
	loop(varname string, counter string, count int) string
	endloop() string
	alu1(varname, from string, op alu1op) string
	alu2(varname, left, right string, op alu2op) string

	footer() string
}

type cLikeRenderer struct {
	depth int
	lang  cLikeLangRenderer
}

func (r *cLikeRenderer) render(irs []*instruction) (*dll, error) {
	/*
	 * devide irs into params and others as their rendering process is different.
	 */

	var params []*instruction

	// pick up params first
	for _, ir := range irs {
		if _, ok := ir.mnemonic.(*mnParam); ok {
			params = append(params, ir)
		}
	}

	// delete params as already picked up above
	irs = slices.DeleteFunc(irs, func(ir *instruction) bool {
		_, ok := ir.mnemonic.(*mnParam)
		return ok
	})

	irmap := map[instid]*instruction{}
	for _, ir := range irs {
		irmap[ir.id] = ir
	}

	/*
	 * start rendering
	 */

	var sb strings.Builder

	write := func(s string) {
		indent := strings.Repeat(r.lang.indent(), r.depth)
		sb.WriteString(indent + s + "\n")
	}

	varname := func(id instid) string {
		if id.valid() {
			return r.lang.varname(int(id))
		}
		return ""
	}

	write(r.lang.header())
	write("")

	// render global params

	for _, param := range params {
		v := varname(param.id)
		write(r.lang.global(v, param.mnemonic.(*mnParam).typ))
	}

	write("")

	toidx := func(id instid) *idx {
		if !id.valid() {
			return nil
		}

		return &idx{val: varname(id)}
	}

	var entry string

	for _, ir := range irs {
		v := varname(ir.id)

		switch mn := ir.mnemonic.(type) {
		case *mnEntry:
			entry = r.lang.entrypointName()
			write(r.lang.entrypoint(entry))
			r.depth++

		case *mnEndEntry:
			r.depth--
			write(r.lang.endEntrypoint())
			write("")

		case *mnKernParam:
			// do nothing

		case *mnKernel:
			kernname := r.lang.kernelName(int(ir.id))

			// extrace kernel parameter
			params := make([]string, len(mn.params))
			typs := make([]typ, len(mn.params))
			for i, paramID := range mn.params {
				param := irmap[paramID]
				params[i] = varname(param.id)
				typs[i] = param.mnemonic.(*mnKernParam).typ // assume param type is *mnKernParam
			}

			write(r.lang.kernel(kernname, params, typs))
			r.depth++

		case *mnEndKernel:
			r.depth--
			write(r.lang.endKernel())
			write("")

		case *mnInvokeKernel:
			kernname := r.lang.kernelName(int(mn.kernel))
			args := make([]string, len(mn.args))
			for i, arg := range mn.args {
				args[i] = varname(arg)
			}
			write(r.lang.invokeKernel(kernname, args))

		case *mnReturn:
			write(r.lang.return_(varname(mn.val)))

		case *mnDecl:
			write(r.lang.decl(v, mn.typ, mn.length))

		case *mnInitImm:
			write(r.lang.initImm(v, mn.val, mn.typ))

		case *mnInit:
			write(r.lang.init_(v, varname(mn.from), toidx(mn.idx)))

		case *mnAssign:
			write(r.lang.assign(varname(mn.left), varname(mn.right), toidx(mn.lidx), toidx(mn.ridx)))

		case *mnLoop:
			write(r.lang.loop(v, varname(mn.count), mn.countImm)) // todo: imm or ref should be handled better
			r.depth++

		case *mnEndLoop:
			r.depth--
			write(r.lang.endloop())
			write("")

		case *mnALU1:
			write(r.lang.alu1(v, varname(mn.val), mn.op))

		case *mnALU2:
			write(r.lang.alu2(v, varname(mn.left), varname(mn.right), mn.op))

		default:
			panic(fmt.Sprintf("unknown mnemonic type: %T", mn))
		}
	}

	write("")

	write(r.lang.footer())

	// finish rendering

	prog := sb.String()
	prog = strings.TrimSpace(prog)

	globs := make([]*dllparam, len(params))
	for i, param := range params {
		globs[i] = &dllparam{name: varname(param.id), value: param.mnemonic.(*mnParam).val.([]float32)}
	}

	return &dll{src: prog, params: globs, entrypoint: entry}, nil
}

/*
 * executor
 */

type executor interface {
	execute(dll *dll) ([]float32, error)
}

func compute(t *Tensor) ([]float32, error) {
	if debug {
		fmt.Println("<tensor AST> ----------------")
		fmt.Println(t)
	}

	var (
		renderer renderer
		executor executor
		gpu      bool
	)

	switch backend {
	case be_golang:
		renderer = &cLikeRenderer{lang: &gorenderer{}}
		executor = &goexecutor{}
		gpu = false

	case be_cuda:
		// renderer = &cLikeRenderer{lang: &cudarenderer{}}
		// executor = &cudaexecutor{}
		// gpu = true
	}

	irs, err := generateIR(t, gpu)
	if err != nil {
		return nil, err
	}

	if debug {
		fmt.Println("<IR list> -------------------")
		for _, ir := range irs {
			fmt.Println(ir)
		}
	}

	// todo: add optimization pass here

	dll, err := renderer.render(irs)
	if err != nil {
		return nil, err
	}

	if debug {
		fmt.Println("<kernel> --------------------")
		fmt.Println(dll.src)

		fmt.Println("<kernel entrypoint> ---------")
		fmt.Println(dll.entrypoint)

		fmt.Println("<kernel parameters> ---------")
		for _, param := range dll.params {
			fmt.Printf("%v: %v\n", param.name, param.value)
		}
	}

	result, err := executor.execute(dll)
	if err != nil {
		return nil, err
	}

	return result, nil

}

func randfilename(pref, ext string) string {
	var letters = []rune("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")
	b := make([]rune, 8)
	for i := range b {
		b[i] = letters[rand.Intn(len(letters))]
	}
	return pref + string(b) + ext
}
