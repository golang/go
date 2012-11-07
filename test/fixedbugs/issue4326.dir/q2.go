package main

import "./q1"

func main() {
      x := 1
      y := q1.Deref(&x)
      if y != 1 {
            panic("y != 1")
      }
}
