package main

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"plugin"
	"strings"
)

/*
 * renderer
 */

var _ cLikeLangRenderer = &gorenderer{}

type gorenderer struct{}

func tostr[T int | float32](vals []T) string {
	s := ""
	for i, val := range vals {
		s += fmt.Sprintf("%v", val)
		if i != len(vals)-1 {
			s += ","
		}
	}
	return s
}

func (r *gorenderer) encodeType(typ typ) string {
	switch typ {
	case t_int:
		return "int"
	case t_ints:
		return "[]int"
	case t_float:
		return "float32"
	case t_floats:
		return "[]float32"
	default:
		panic("unknown typ!")
	}
}

func (r *gorenderer) encodeImm(typ typ, imm any) string {
	s := ""
	switch typ {
	case t_int:
		s = fmt.Sprintf("%v", imm.(int))

	case t_ints:
		s = fmt.Sprintf("[]int{%v}", tostr(imm.([]int)))

	case t_float:
		s = fmt.Sprintf("%v", imm.(float32))

	case t_floats:
		s = fmt.Sprintf("[]float32{%v}", tostr(imm.([]float32)))
	}

	return s
}

func (r *gorenderer) indent() string {
	return "	"
}

func (r *gorenderer) header() string {
	return "package main"
}

func (r *gorenderer) varname(id int) string {
	return fmt.Sprintf("D%v", id)
}

func (r *gorenderer) entrypoint() string {
	return "F" // must be exported
}

func (r *gorenderer) kernel(entry string) string {
	return fmt.Sprintf("func %v() []float32 {", entry)
}

func (r *gorenderer) endkernel() string {
	return "}"
}

func (r *gorenderer) global(varname string, typ typ) string {
	return fmt.Sprintf("var %v %v", varname, r.encodeType(typ))
}

func (r *gorenderer) return_(varname string) string {
	return fmt.Sprintf("return %v", varname)
}

func (r *gorenderer) decl(varname string, typ typ, length int) string {
	t := r.encodeType(typ)
	if typ.isarray() {
		return fmt.Sprintf("var %v %v = make(%v, %v)", varname, t, t, length)
	}

	return fmt.Sprintf("var %v %v", varname, t)
}

func (r *gorenderer) initImm(varname string, imm any, typ typ) string {
	return fmt.Sprintf("var %v %v = %v", varname, r.encodeType(typ), r.encodeImm(typ, imm))
}

func (r *gorenderer) init_(varname string, from string, idx *idx) string {
	if idx != nil {
		return fmt.Sprintf("var %v = %v[%v]", varname, from, idx.val)
	}

	return fmt.Sprintf("var %v = %v", varname, from)
}

func (r *gorenderer) assign(left, right string, lidx, ridx *idx) string {
	if lidx != nil && ridx != nil {
		return fmt.Sprintf("%v[%v] = %v[%v]", left, lidx.val, right, ridx.val)
	}

	if lidx != nil {
		return fmt.Sprintf("%v[%v] = %v", left, lidx.val, right)
	}

	if ridx != nil {
		return fmt.Sprintf("%v = %v[%v]", left, right, ridx.val)
	}

	return fmt.Sprintf("%v = %v", left, right)
}

func (r *gorenderer) loop(varname string, counter string, count int) string {
	cnt := counter
	if cnt == "" {
		cnt = fmt.Sprintf("%d", count)
	}

	return fmt.Sprintf("for %v := 0; %v < %v; %v++ {", varname, varname, cnt, varname)
}

func (r *gorenderer) endloop() string {
	return "}"
}

func (r *gorenderer) alu1(varname, from string, op alu1op) string {
	switch op {
	case alu1_neg:
		return fmt.Sprintf("%v = -%v", varname, from)
	default:
		panic("unknown alu1op!")
	}
}

func (r *gorenderer) alu2(varname, left, right string, op alu2op) string {
	switch op {
	case alu2_add:
		return fmt.Sprintf("var %v = %v + %v", varname, left, right)
	case alu2_mul:
		return fmt.Sprintf("var %v = %v * %v", varname, left, right)
	default:
		panic("unknown alu2op!")
	}
}

func (r *gorenderer) footer() string {
	return ""
}

/*
 * executor
 */

type goexecutor struct{}

func (e *goexecutor) execute(dll *dll) ([]float32, error) {
	// create source file
	dir := os.TempDir()

	// full path
	filename := filepath.Join(dir, randfilename("whale_go_", ".go"))

	err := os.WriteFile(filename, []byte(dll.src), 0666)
	if err != nil {
		return nil, err
	}

	// compile so
	basename := strings.TrimSuffix(filename, filepath.Ext(filename))
	soname := basename + ".so"
	out, err := exec.Command("go", "build", "-o", soname, "-buildmode=plugin", filename).CombinedOutput()
	if err != nil {
		return nil, fmt.Errorf("cannot compile so (%v): %v", err, string(out))
	}

	// call so

	// note that in Go opened plugin cannot be dlclosed
	p, err := plugin.Open(soname)
	if err != nil {
		return nil, fmt.Errorf("goexecutor: cannot open plugin %v: %v", soname, err)
	}

	for _, param := range dll.params {
		d, err := p.Lookup(param.name)
		if err != nil {
			return nil, fmt.Errorf("goexecutor: cannot lookup symbol %v: %v", param.name, err)
		}

		*d.(*[]float32) = param.value
	}

	entry, err := p.Lookup(dll.entrypoint)
	if err != nil {
		return nil, fmt.Errorf("goexecutor: cannot lookup symbol %v: %v", dll.entrypoint, err)
	}

	result := entry.(func() []float32)()
	return result, nil
}
