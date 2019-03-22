package reflect

type Type interface {
	String() string
}

type Value struct {
}

func (Value) String() string

func SliceOf(Type) Type

func TypeOf(interface{}) Type

func ValueOf(interface{}) Value
