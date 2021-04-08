package genUtils


import "fmt"
import "os"

var ParamFailCount int

var ReturnFailCount int

var FailCount int

var Mode string

type UtilsType int

//go:noinline
func NoteFailure(cm int, pidx int, fidx int, pkg string, pref string, parmNo int, isret bool,_ uint64) {
  if isret {
    if ParamFailCount != 0 {
      return
    }
    ReturnFailCount++
  } else {
    ParamFailCount++
  }
  fmt.Fprintf(os.Stderr, "Error: fail %s |%d|%d|%d| =%s.Test%d= %s %d\n", Mode, cm, pidx, fidx, pkg, fidx, pref, parmNo)

  if (ParamFailCount + FailCount + ReturnFailCount > 9999) {
    os.Exit(1)
  }
}

//go:noinline
func NoteFailureElem(cm int, pidx int, fidx int, pkg string, pref string, parmNo int, elem int, isret bool, _ uint64) {

  if isret {
    if ParamFailCount != 0 {
      return
    }
    ReturnFailCount++
  } else {
    ParamFailCount++
  }
  fmt.Fprintf(os.Stderr, "Error: fail %s |%d|%d|%d| =%s.Test%d= %s %d elem %d\n", Mode, cm, pidx, fidx, pkg, fidx, pref, parmNo, elem)

  if (ParamFailCount + FailCount + ReturnFailCount > 9999) {
    os.Exit(1)
  }
}

func BeginFcn() {
  ParamFailCount = 0
  ReturnFailCount = 0
}

func EndFcn() {
  FailCount += ParamFailCount
  FailCount += ReturnFailCount
}

