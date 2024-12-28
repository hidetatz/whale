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
 * irgenerator: generate a list of IR from AST (tensor computation graph).
 */

type irgenerator interface {
	generate(t *Tensor) ([]*instruction, error)
}

/*
 * Generator for basic machine
 */

type basegenerator struct {
	override func(t *Tensor) instid
}

func (g *basegenerator) generate(t *Tensor) (irs []*instruction, err error) {
	irs = []*instruction{}

	defer func() {
		if r := recover(); r != nil {
			err = fmt.Errorf(r.(string))
			irs = nil
		}
	}()

	push := func(ir *instruction) instid {
		irs = append(irs, ir)
		return ir.id
	}

	var dfs func(t *Tensor) instid
	dfs = func(t *Tensor) instid {
		if g.override != nil {
			overriddenID := g.override(t)
			if overriddenID.valid() {
				return overriddenID
			}
		}

		switch t.op {
		case op_const:
			return push(inst(&mnParam{typ: t_floats, val: t.data}))

		case op_add, op_mul:
			// todo: scalar optimization

			l, r := t.inputs[0], t.inputs[1]
			sizel, sizer := l.Size(), r.Size()
			if sizel != sizer {
				panic(fmt.Sprintf("cannot compute 2 different size tensors: %v and %v", l, r))
			}

			// get left and right id first
			lid, rid := dfs(l), dfs(r)

			// define result to store
			result := push(inst(&mnDecl{typ: t_floats, length: sizel}))

			// start loop
			loop := push(inst(&mnLoop{countImm: sizel}))

			// assume vector
			// todo: support 2 or more dimensions

			// compute stride, considering broadcast
			lstride := push(inst(&mnInitImm{typ: t_int, val: l.dim.strides[0]}))
			rstride := push(inst(&mnInitImm{typ: t_int, val: r.dim.strides[0]}))

			// define index
			lidx := push(inst(&mnALU2{left: loop, op: alu2_mul, right: lstride}))
			ridx := push(inst(&mnALU2{left: loop, op: alu2_mul, right: rstride}))

			// load value to be computed from left and right
			loadl := push(inst(&mnInit{from: lid, idx: lidx}))
			loadr := push(inst(&mnInit{from: rid, idx: ridx}))

			var op alu2op

			if t.op == op_add {
				op = alu2_add
			} else {
				op = alu2_mul
			}

			// do compute
			alu2 := push(inst(&mnALU2{left: loadl, op: op, right: loadr}))

			// assign computed to result
			push(inst(&mnAssign{left: result, lidx: loop, right: alu2}))

			// finish loop
			push(inst(&mnEndLoop{}))

			return result

		case op_expand:
			return dfs(t.inputs[0])

		default:
			panic(fmt.Sprintf("unknown op: %v", t.op))
		}
	}

	result := dfs(t)
	push(inst(&mnReturn{val: result}))

	return irs, nil
}

/*
 * Generator for CPU machine.
 * The same as base.
 */

type cpuGenerator struct{}

func (g *cpuGenerator) generate(t *Tensor) (irs []*instruction, err error) {
	base := &basegenerator{}
	return base.generate(t)
}

/*
 * Generator for GPU machine.
 * Mostly the same as base, but optimized for parallel computation.
 */

type gpuGenerator struct{}

func (g *gpuGenerator) generate(t *Tensor) (irs []*instruction, err error) {
	// todo: override
	base := &basegenerator{}
	return base.generate(t)
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

	entrypoint() string

	kernel(entry string) string
	endkernel() string

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

	// render main kernel

	entry := r.lang.entrypoint()
	write(r.lang.kernel(entry))
	r.depth++

	toidx := func(id instid) *idx {
		if !id.valid() {
			return nil
		}

		return &idx{val: varname(id)}
	}

	for _, ir := range irs {
		v := varname(ir.id)

		switch mn := ir.mnemonic.(type) {
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

		case *mnALU1:
			write(r.lang.alu1(v, varname(mn.val), mn.op))

		case *mnALU2:
			write(r.lang.alu2(v, varname(mn.left), varname(mn.right), mn.op))

		default:
			panic(fmt.Sprintf("unknown mnemonic type: %T", mn))
		}
	}

	r.depth--
	write(r.lang.endkernel())
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
		irgenerator irgenerator
		renderer    renderer
		executor    executor
	)

	switch backend {
	case be_golang:
		irgenerator = &cpuGenerator{}
		renderer = &cLikeRenderer{lang: &gorenderer{}}
		executor = &goexecutor{}

	case be_cuda:
		// irgenerator = &gpuGenerator{}
		// renderer = &cLikeRenderer{lang: &cudarenderer{}}
		// executor = &cudaexecutor{}
	}

	irs, err := irgenerator.generate(t)
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
