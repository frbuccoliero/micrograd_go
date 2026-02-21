package engine

import (
	"errors"
)

type Value struct{
	Data float32
	Grad float32	
}

func NewValue(data float32) Value {
	return Value{Data: data, Grad: 0.0}
}

func ValidateOperand(op interface{}) (Value, error){
	_op, ok_op := op.(Value)
	if !ok_op {
		_op_val, ok_op_val := op.(float32)
		if !ok_op_val {
			return NewValue(0), errors.New("Can't assert type of operand. Enter a Value or float32")
		} else{
			return NewValue(_op_val), nil
		}
	}
	return _op, nil
}

func Add(a interface{}, b interface{}) Value {

	_a, _ := ValidateOperand(a)
	_b, _ := ValidateOperand(b)

	out := NewValue(_a.Data + _b.Data)
	return out
}
