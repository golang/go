//go:build js && wasm

package js

// Promisable function that satisfies MakePromise's requirements.
type PromiseAbleFunc = func(Value, []Value) (interface{}, error)

// MakePromise makes a promise of a function that takes an array of arguments.
func PromiseOf(fn PromiseAbleFunc) Func {
	return FuncOf(func(this Value, args []Value) interface{} {
		// Handler for this Promise.
		handler := FuncOf(func(promiseThis Value, promiseArgs []Value) interface{} {
			resolve := promiseArgs[0]
			reject := promiseArgs[1]

			// Run this asynchronously.
			go func() {
				res, err := fn(this, args)
				if err != nil {
					errorConstructor := Global().Get("Error")
					errorObject := errorConstructor.New(err.Error())
					reject.Invoke(errorObject)
					return
				}
				resolve.Invoke(res)
			}()

			// Handler doesn't return anything directly.
			return nil
		})

		// Create the Promise and return.
		promiseConstructor := Global().Get("Promise")
		return promiseConstructor.New(handler)
	})
}
