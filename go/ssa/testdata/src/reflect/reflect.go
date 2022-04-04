package reflect

type Type interface {
	Elem() Type
	Kind() Kind
	String() string
}

type Value struct{}

func (Value) String() string
func (Value) Elem() Value
func (Value) Field(int) Value
func (Value) Index(i int) Value
func (Value) Int() int64
func (Value) Interface() interface{}
func (Value) IsNil() bool
func (Value) IsValid() bool
func (Value) Kind() Kind
func (Value) Len() int
func (Value) MapIndex(Value) Value
func (Value) MapKeys() []Value
func (Value) NumField() int
func (Value) Pointer() uintptr
func (Value) SetInt(int64)
func (Value) Type() Type

func SliceOf(Type) Type
func TypeOf(interface{}) Type
func ValueOf(interface{}) Value

type Kind uint

const (
	Invalid Kind = iota
	Int
	Pointer
)

func DeepEqual(x, y interface{}) bool
