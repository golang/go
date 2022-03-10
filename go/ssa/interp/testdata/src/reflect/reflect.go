package reflect

type Type interface {
	String() string
	Kind() Kind
	Elem() Type
}

type Value struct {
}

func (Value) String() string

func (Value) Elem() Value
func (Value) Kind() Kind
func (Value) Int() int64
func (Value) IsValid() bool
func (Value) IsNil() bool
func (Value) Len() int
func (Value) Pointer() uintptr
func (Value) Index(i int) Value
func (Value) Type() Type
func (Value) Field(int) Value
func (Value) MapIndex(Value) Value
func (Value) MapKeys() []Value
func (Value) NumField() int
func (Value) Interface() interface{}

func SliceOf(Type) Type

func TypeOf(interface{}) Type

func ValueOf(interface{}) Value

type Kind uint

// Constants need to be kept in sync with the actual definitions for comparisons in tests.
const (
	Invalid Kind = iota
	Bool
	Int
	Int8
	Int16
	Int32
	Int64
	Uint
	Uint8
	Uint16
	Uint32
	Uint64
	Uintptr
	Float32
	Float64
	Complex64
	Complex128
	Array
	Chan
	Func
	Interface
	Map
	Pointer
	Slice
	String
	Struct
	UnsafePointer
)

const Ptr = Pointer
