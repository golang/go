# Section 7: eject lowering templates (archived)

Historical, non-normative text from logos-docs/PLAN.md lines 374-498 at commit 57475ed. The normative spec is logos-docs/SPEC.md; the build plan is logos-docs/IMPLEMENTATION.md.

**SECTION 7: EXACT `--emit-go` TRANSPILATION LOWERING TEMPLATES (`src/cmd/compile/internal/eject`):**

Every `loGo/s` construct has a deterministic 1:1 pure-Go AST lowering rule:

**7.1. `errdefer` + `defer fn()?` + `?` Propagation Lowering:**
* **Input (`transaction.lgo`):**
  ```go
  func Transfer(db *DB, from int64, to int64, amt int64) (int64, error) {
      tx := db.Begin()?
      errdefer tx.Rollback()?
      defer tx.Close()?

      bal := tx.AddBalance(from, -amt)?
      tx.AddBalance(to, amt)?
      tx.Commit()?
      return bal, nil
  }
  ```
* **Emitted Pure Go (`transaction.gen.go`):**
  ```go
  //go:build !logos
  package bank

  import _logos_errors "errors"

  func Transfer(db *DB, from int64, to int64, amt int64) (_ret0 int64, _retErr error) {
      _errdefer_ok := false
      tx, _err0 := db.Begin()
      if _err0 != nil {
          return _ret0, _err0
      }
      defer func() {
          if !_errdefer_ok || (_retErr != nil) {
              if _rbErr := tx.Rollback(); _rbErr != nil {
                  if _retErr == nil {
                      _retErr = _rbErr
                  } else {
                      _retErr = _logos_errors.Join(_retErr, _rbErr)
                  }
              }
          }
      }()
      defer func() {
          if _clErr := tx.Close(); _clErr != nil {
              if _retErr == nil {
                  _retErr = _clErr
              } else {
                  _retErr = _logos_errors.Join(_retErr, _clErr)
              }
          }
      }()
      bal, _err1 := tx.AddBalance(from, -amt)
      if _err1 != nil {
          return _ret0, _err1
      }
      _, _err2 := tx.AddBalance(to, amt)
      if _err2 != nil {
          return _ret0, _err2
      }
      if _err3 := tx.Commit(); _err3 != nil {
          return _ret0, _err3
      }
      _errdefer_ok = true
      return bal, nil
  }
  ```

**7.2. `const switch T` Generic Type Switch Lowering:**
* **Input (`codec.lgo`):**
  ```go
  func EncodeSlice[T any](buf []byte, items const []T) ([]byte, error) {
      const switch T {
      case int64:
          for _, v := range items {
              buf = strconv.AppendInt(buf, v, 10)
          }
          return buf, nil
      case io.WriterTo:
          // Constraint intersection: T keeps concrete layout, gains WriteTo
          var b bytes.Buffer
          for _, v := range items {
              v.WriteTo(&b)?
          }
          return append(buf, b.Bytes()...), nil
      }
  }
  ```
* **Emitted Pure Go (`codec.gen.go`):**
  ```go
  //go:build !logos
  package codec

  import (
      "bytes"
      "io"
      "strconv"
      _logos_unsafe "unsafe"
  )

  func EncodeSlice[T any](buf []byte, items []T) ([]byte, error) {
      switch any((*T)(nil)).(type) {
      case *int64:
          _items := *(*[]int64)(_logos_unsafe.Pointer(&items))
          for _, v := range _items {
              buf = strconv.AppendInt(buf, v, 10)
          }
          return buf, nil
      default:
          if _, _ok := any(*new(T)).(io.WriterTo); _ok {
              var b bytes.Buffer
              for _, v := range items {
                  _, _err0 := any(v).(io.WriterTo).WriteTo(&b)
                  if _err0 != nil {
                      return nil, _err0
                  }
              }
              return append(buf, b.Bytes()...), nil
          }
          return buf, nil
      }
  }
  ```

---

