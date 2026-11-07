//go:build js && wasm

package js

import "errors"

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
				// Recover by rejecting.
				defer func() {
					if err := recover(); err != nil {
						reject.Invoke(PromiseError(err))
					}
				}()			
				
				// Run the PromiseAble func with original this and args.
				res, err := fn(this, args)
				if err != nil {
					reject.Invoke(PromiseError(err))
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

// PromiseError makes sure to return some error that Invoke will understand.
func PromiseError(e interface{}) (err error) {
	switch x := e.(type) {
	case string:
		err = errors.New(x)
	case error:
		err = x
	default:
		err = errors.New("unknown panic")
	}
	return err
}
