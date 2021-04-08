package genCaller42

import "bad_select_crash.dir/genChecker42"
import "bad_select_crash.dir/genUtils"
import "reflect"


func Caller2() {
  genUtils.BeginFcn()
  c0 := genChecker42.StructF2S0{F0: genChecker42.ArrayF2S1E1{genChecker42.New_3(float64(-0.4418990509835844))}}
  c1 := genChecker42.ArrayF2S2E1{genChecker42.StructF2S1{/* _: "񊶿(z̽|" */F1: "􂊇񊶿"}}
  c2 := int16(4162)
  c3 := float32(-7.667096e+37)
  c4 := int64(3202175648847048679)
  var p0 genChecker42.ArrayF2S0E0
  p0 = genChecker42.ArrayF2S0E0{}
  var p1 uint8
  p1 = uint8(57)
  var p2 uint16
  p2 = uint16(10920)
  var p3 float64
  p3 = float64(-1.597256501942112)
  genUtils.Mode = ""
  // 5 returns 4 params
  r0, r1, r2, r3, r4 := genChecker42.Test2(p0, p1, p2, p3)
  if !genChecker42.EqualStructF2S0(r0, c0) {
    genUtils.NoteFailure(9, 42, 2, "genChecker42", "return", 0, true, uint64(0))
  }
  if r1 != c1 {
    genUtils.NoteFailure(9, 42, 2, "genChecker42", "return", 1, true, uint64(0))
  }
  if r2 != c2 {
    genUtils.NoteFailure(9, 42, 2, "genChecker42", "return", 2, true, uint64(0))
  }
  if r3 != c3 {
    genUtils.NoteFailure(9, 42, 2, "genChecker42", "return", 3, true, uint64(0))
  }
  if r4 != c4 {
    genUtils.NoteFailure(9, 42, 2, "genChecker42", "return", 4, true, uint64(0))
  }
  // same call via reflection
  genUtils.Mode = "reflect"
  rc := reflect.ValueOf(genChecker42.Test2)
  rvslice :=   rc.Call([]reflect.Value{reflect.ValueOf(p0), reflect.ValueOf(p1), reflect.ValueOf(p2), reflect.ValueOf(p3)})
  rr0i := rvslice[0].Interface()
  rr0v:= rr0i.( genChecker42.StructF2S0)
  if !genChecker42.EqualStructF2S0(rr0v, c0) {
    genUtils.NoteFailure(9, 42, 2, "genChecker42", "return", 0, true, uint64(0))
  }
  rr1i := rvslice[1].Interface()
  rr1v:= rr1i.( genChecker42.ArrayF2S2E1)
  if rr1v != c1 {
    genUtils.NoteFailure(9, 42, 2, "genChecker42", "return", 1, true, uint64(0))
  }
  rr2i := rvslice[2].Interface()
  rr2v:= rr2i.( int16)
  if rr2v != c2 {
    genUtils.NoteFailure(9, 42, 2, "genChecker42", "return", 2, true, uint64(0))
  }
  rr3i := rvslice[3].Interface()
  rr3v:= rr3i.( float32)
  if rr3v != c3 {
    genUtils.NoteFailure(9, 42, 2, "genChecker42", "return", 3, true, uint64(0))
  }
  rr4i := rvslice[4].Interface()
  rr4v:= rr4i.( int64)
  if rr4v != c4 {
    genUtils.NoteFailure(9, 42, 2, "genChecker42", "return", 4, true, uint64(0))
  }
  genUtils.EndFcn()
}
