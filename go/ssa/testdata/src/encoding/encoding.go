package encoding

type BinaryMarshaler interface {
	MarshalBinary() (data []byte, err error)
}

type BinaryUnmarshaler interface {
	UnmarshalBinary(data []byte) error
}
