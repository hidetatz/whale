package main

import (
	"fmt"
)

/*
 * instruction ID
 */

type instid int

var curinstid instid = 0

func newInstid() instid {
	curinstid++
	return curinstid
}

func (id instid) String() string {
	return fmt.Sprintf("_%v", int(id)) // to tell it's not an immediate value
}

func (id instid) valid() bool {
	return id > 0
}

/*
 * instruction
 */

type instruction struct {
	id       instid
	mnemonic mnemonic
}

func (i *instruction) String() string {
	return fmt.Sprintf("%v: %v", i.id, i.mnemonic)
}

func inst(m mnemonic) *instruction {
	return &instruction{id: newInstid(), mnemonic: m}
}

/*
 * mnemonics
 */

type typ int

const (
	t_int typ = iota + 1
	t_ints
	t_float
	t_floats
)

func (t typ) isarray() bool {
	return t == t_ints || t == t_floats
}

func (t typ) String() string {
	switch t {
	case t_int:
		return "int"
	case t_ints:
		return "ints"
	case t_float:
		return "float"
	case t_floats:
		return "floats"
	default:
		panic("unknown typ")
	}
}

type mnemonic interface {
	ismnemonic()
	fmt.Stringer
}

type mnParam struct {
	mnemonic

	typ typ
	val any
}

type mnReturn struct {
	mnemonic

	val instid
}

func (m *mnReturn) String() string {
	return fmt.Sprintf("{return %v}", m.val)
}

func (m *mnParam) String() string {
	return fmt.Sprintf("{param %v (%v)}", m.val, m.typ)
}

// declare a variable without initialization
// var a;
type mnDecl struct {
	mnemonic

	typ    typ
	length int
}

func (m *mnDecl) String() string {
	if m.typ.isarray() {
		return fmt.Sprintf("{decl %v (len=%v)}", m.typ, m.length)
	}
	return fmt.Sprintf("{decl %v}", m.typ)
}

// declare and initialize a variable with immediate value
// var a = [1, 2, 3];
type mnInitImm struct {
	mnemonic

	typ typ
	val any
}

func (m *mnInitImm) String() string {
	return fmt.Sprintf("{init_imm %v (%v)}", m.val, m.typ)
}

// declare and initialize a variable with already-defined value
// var a = b;
// var a = b[i];
type mnInit struct {
	mnemonic

	from instid
	idx  instid // optional
}

func (m *mnInit) String() string {
	if m.idx.valid() {
		return fmt.Sprintf("{init %v[%v]}", m.from, m.idx)
	}
	return fmt.Sprintf("{init %v}", m.from)
}

// assign an already-defined value to other variable
// a = b;
// a[i] = b;
// a = b[i];
// a[i] = b[j];
type mnAssign struct {
	mnemonic

	left  instid
	lidx  instid
	right instid
	ridx  instid
}

func (m *mnAssign) String() string {
	if m.lidx.valid() && m.ridx.valid() {
		return fmt.Sprintf("{assign %v[%v] = %v[%v]}", m.left, m.lidx, m.right, m.ridx)
	}

	if m.lidx.valid() {
		return fmt.Sprintf("{assign %v[%v] = %v}", m.left, m.lidx, m.right)
	}

	if m.ridx.valid() {
		return fmt.Sprintf("{assign %v = %v[%v]}", m.left, m.lidx, m.right)
	}

	return fmt.Sprintf("{assign %v = %v}", m.left, m.right)
}

// start loop
// for (var i = 0; i < cnt; i++) {
type mnLoop struct {
	mnemonic

	count    instid
	countImm int
}

func (m *mnLoop) String() string {
	if m.count.valid() {
		return fmt.Sprintf("{loop i in 0..%v}", m.count)
	}
	return fmt.Sprintf("{loop i in 0..%v}", m.countImm)
}

// ends loop
type mnEndLoop struct {
	mnemonic
}

func (m *mnEndLoop) String() string {
	return fmt.Sprintf("{endloop}")
}

type alu1op int

const (
	alu1_neg alu1op = iota + 1
)

// arithmetic operation with 1 operand.
// f(a)
type mnALU1 struct {
	mnemonic

	op  alu1op
	val instid
}

func (m *mnALU1) String() string {
	return fmt.Sprintf("{alu1 %v %v}", m.op, m.val)
}

type alu2op int

const (
	alu2_add alu2op = iota + 1
	alu2_mul
)

// arithmetic operation with 2 operands.
// f(a, b)
type mnALU2 struct {
	mnemonic

	op    alu2op
	left  instid
	right instid
}

func (m *mnALU2) String() string {
	return fmt.Sprintf("{alu2 %v %v %v}", m.left, m.op, m.right)
}
