package q1

func Deref(typ interface{}) interface{} {
      if typ, ok := typ.(*int); ok {
            return *typ
      }
      return typ
}
