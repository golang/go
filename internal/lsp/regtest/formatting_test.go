package regtest

import (
	"context"
	"testing"
)

const unformattedProgram = `
-- main.go --
package main
import "fmt"
func main(  ) {
	fmt.Println("Hello World.")
}
-- main.go.golden --
package main

import "fmt"

func main() {
	fmt.Println("Hello World.")
}
`

func TestFormatting(t *testing.T) {
	runner.Run(t, unformattedProgram, func(ctx context.Context, t *testing.T, env *Env) {
		env.OpenFile("main.go")
		env.FormatBuffer("main.go")
		got := env.E.BufferText("main.go")
		want := env.ReadWorkspaceFile("main.go.golden")
		if got != want {
			t.Errorf("\n## got formatted file:\n%s\n## want:\n%s", got, want)
		}
	})
}

const disorganizedProgram = `
-- main.go --
package main

import (
	"fmt"
	"errors"
)
func main(  ) {
	fmt.Println(errors.New("bad"))
}
-- main.go.organized --
package main

import (
	"errors"
	"fmt"
)
func main(  ) {
	fmt.Println(errors.New("bad"))
}
-- main.go.formatted --
package main

import (
	"errors"
	"fmt"
)

func main() {
	fmt.Println(errors.New("bad"))
}
`

func TestOrganizeImports(t *testing.T) {
	runner.Run(t, disorganizedProgram, func(ctx context.Context, t *testing.T, env *Env) {
		env.OpenFile("main.go")
		env.OrganizeImports("main.go")
		got := env.E.BufferText("main.go")
		want := env.ReadWorkspaceFile("main.go.organized")
		if got != want {
			t.Errorf("\n## got formatted file:\n%s\n## want:\n%s", got, want)
		}
	})
}

func TestFormattingOnSave(t *testing.T) {
	runner.Run(t, disorganizedProgram, func(ctx context.Context, t *testing.T, env *Env) {
		env.OpenFile("main.go")
		env.SaveBuffer("main.go")
		got := env.E.BufferText("main.go")
		want := env.ReadWorkspaceFile("main.go.formatted")
		if got != want {
			t.Errorf("\n## got formatted file:\n%s\n## want:\n%s", got, want)
		}
	})
}
