# It's MyGO!!!!!

[中文文档 (Chinese Version)](README_CN.md)

This repository is an unofficial personal modification (fork) of GO, hence the name `MyGO`.

![MyGO image](img/mygo1.png)
*MyGO image by Gemini/Nano-banana-pro*

![MyGO Logo](img/mygo2.jpg)
*MyGO Logo*

--------------

## Design Principles and Core Scenarios

- ⚡️ **Intuitive Vector Calculation**: MyGO introduces native syntax support for vector calculations that aligns with mathematical intuition. This design aligns code logic closely with mathematical formulas, significantly improving the readability and maintainability of linear algebra operations.

- 🚀 **Minimalism and Efficiency**: Pursuing Python-like extreme conciseness and expressiveness. Through rich syntactic sugar, it drastically reduces boilerplate code, enhancing development efficiency while ensuring runtime overhead is minimized, balancing performance and elegance.

## 1. Decorators

Decorators allow you to wrap functions using the `@decorator` syntax at the function declaration.

### 1.1 Basic Decorators (No-Argument Functions)

```go
func logger(f func()) func() {
    return func() {
        fmt.Println("Start execution")
        f()
        fmt.Println("Execution finished")
    }
}

@logger
func sayHello() {
    fmt.Println("Hello, MyGO!")
}

func main() {
    sayHello()
    // Output:
    // Start execution
    // Hello, MyGO!
    // Execution finished
}
```

### 1.2 Decorating Functions with Arguments

Decorators can wrap functions with any signature, including those with parameters and return values.

```go
// Timing decorator - decorates functions with args and return values
func timeit(f func(int, int) int) func(int, int) int {
    return func(a, b int) int {
        start := time.Now()
        result := f(a, b)
        elapsed := time.Since(start)
        fmt.Printf("Execution time: %v\n", elapsed)
        return result
    }
}

@timeit
func add(x, y int) int {
    time.Sleep(100 * time.Millisecond)
    return x + y
}

func main() {
    result := add(3, 5)
    fmt.Println("Result:", result)
    // Output:
    // Execution time: 100.xxxms
    // Result: 8
}
```

### 1.3 Decorators with Arguments

The decorator itself can accept arguments to implement more flexible logic.

```go
// Decorator with arguments: repeat n times
func repeat(f func(), n int) func() {
    return func() {
        for i := 0; i < n; i++ {
            fmt.Printf("Execution #%d:\n", i+1)
            f()
        }
    }
}

@repeat(3)
func greet() {
    fmt.Println("Hello, MyGO!")
}

func main() {
    greet()
    // Output:
    // Execution #1:
    // Hello, MyGO!
    // ...
    // Execution #3:
    // Hello, MyGO!
}
```

### 1.4 Chaining Multiple Decorators

You can apply multiple decorators to a single function; they are executed from top to bottom.

```go
func logger(f func()) func() {
    return func() {
        fmt.Println("[LOG] Start")
        f()
        fmt.Println("[LOG] End")
    }
}

func uppercase(f func()) func() {
    return func() {
        fmt.Println("=== START ===")
        f()
        fmt.Println("=== END ===")
    }
}

@logger
@uppercase
func sayHello() {
    fmt.Println("Hello, MyGO!")
}

func main() {
    sayHello()
    // Output:
    // [LOG] Start
    // === START ===
    // Hello, MyGO!
    // === END ===
    // [LOG] End
}
```

### 1.5 Error Handling Decorators

```go
// Error recovery decorator
func recover_errors(f func()) func() {
    return func() {
        defer func() {
            if r := recover(); r != nil {
                fmt.Printf("Caught error: %v\n", r)
            }
        }()
        f()
    }
}

@recover_errors
func riskyOperation() {
    fmt.Println("Performing risky operation...")
    panic("Something went wrong!")
}

func main() {
    riskyOperation()
    fmt.Println("Program continues")
    // Output:
    // Performing risky operation...
    // Caught error: Something went wrong!
    // Program continues
}
```

### 1.6 Authentication Decorators

```go
var currentUser = "admin"

func requireAuth(f func(string), role string) func(string) {
    return func(user string) {
        if user != role {
            fmt.Printf("Access denied: requires %s role\n", role)
            return
        }
        f(user)
    }
}

@requireAuth("admin")
func deleteUser(user string) {
    fmt.Printf("User %s deleted\n", user)
}

func main() {
    deleteUser("admin") // Output: User admin deleted
    deleteUser("guest") // Output: Access denied: requires admin role
}
```

### 1.7 Best Practices for Decorators with Varying Parameter Counts

You can use `reflect` to achieve this.

```go
package main

import (
    "fmt"
    "reflect"
    "time"
)

// Generic timing decorator
// T can be any function type
func TimeIt[T any](f T) T {
    // 1. Get reflection value and type of the function
    fnVal := reflect.ValueOf(f)
    fnType := fnVal.Type()

    // Ensure it is a function
    if fnType.Kind() != reflect.Func {
        panic("TimeIt decorator requires a function")
    }

    // 2. Use MakeFunc to create a new function
    wrapper := reflect.MakeFunc(fnType, func(args []reflect.Value) []reflect.Value {
        start := time.Now()

        // 3. Call the original function
        // Note: Reflection has some performance overhead, usually negligible in business logic
        results := fnVal.Call(args)

        elapsed := time.Since(start)
        fmt.Printf("==> Execution time: %v\n", elapsed)

        return results
    })

    // 4. Convert the created reflection value back to type T
    return wrapper.Interface().(T)
}

// Decorator: 2 args
@TimeIt
func add(x, y int) int {
    time.Sleep(50 * time.Millisecond)
    return x + y
}

// Decorator: 1 arg
@TimeIt
func inverse(x int) int {
    time.Sleep(50 * time.Millisecond)
    return 100 / x
}

// Decorator: 0 args, void
@TimeIt
func sayHello() {
    fmt.Println("Hello MyGO!")
}

func main() {
    fmt.Println("--- Test 1: Add ---")
    r1 := add(3, 5)
    fmt.Println("Result:", r1)

    fmt.Println("\n--- Test 2: Inverse ---")
    r2 := inverse(2)
    fmt.Println("Result:", r2)

    fmt.Println("\n--- Test 3: Void ---")
    sayHello()
}
```

## 2. Default Function Arguments

Supports setting default values for function parameters. Arguments with default values can be omitted during calls.

**Rule**: Default values can only be set from back to front (i.e., parameters with default values must be at the end of the parameter list).

```go
// All parameters have defaults
func greet(name string = "World", greeting string = "Hello") {
    fmt.Printf("%s, %s!\n", greeting, name)
}

// Partial defaults
func calculate(x int, y int = 10, z int = 5) int {
    return x + y + z
}

func main() {
    greet()             // Output: Hello, World!
    greet("MyGO")       // Output: Hello, MyGO!
    greet("MyGO", "Hi") // Output: Hi, MyGO!

    fmt.Println(calculate(1))       // Output: 16 (1 + 10 + 5)
    fmt.Println(calculate(1, 2))    // Output: 8 (1 + 2 + 5)
    fmt.Println(calculate(1, 2, 3)) // Output: 6 (1 + 2 + 3)
}
```

## 3. Optional Chaining

Use the `?.` operator for null-safe field access and method calls, avoiding nil pointer panics.

*Important! Optional chaining always returns a **pointer type** and must be dereferenced for value types! — This is to support nil.*

```go
type User struct {
    Name    string
    Profile *Profile
}

type Profile struct {
    Email string
    Age   int
}

func main() {
    user1 := &User{Name: "Alice", Profile: &Profile{Email: "alice@example.com", Age: 25}}
    user2 := &User{Name: "Bob", Profile: nil}

    // Traditional way requires nil check
    if user2.Profile != nil {
        fmt.Println(user2.Profile.Email)
    }

    // Using optional chaining, handles nil automatically
    email1 := user1?.Profile?.Email // "alice@example.com" (type *string)
    email2 := user2?.Profile?.Email // nil (wont panic)

    fmt.Println("email1:", *email1) // alice@example.com
    fmt.Println("email2:", email2)  // <nil>
}
```

## 4. Ternary Expression

Supports concise conditional expressions `condition ? trueValue : falseValue`.

**Note**: The True branch and False branch of a ternary expression must guarantee the **same type**, otherwise an error occurs!

```go
func main() {
    age := 18

    // Traditional if-else
    var status string
    if age >= 18 {
        status = "Adult"
    } else {
        status = "Minor"
    }

    status = age >= 18 ? "Adult" : "Minor"
    fmt.Println(status) // Output: Adult

    fmt.Println(age >= 18 ? "Adult" : "Minor") // Output: Adult

    // Nested ternary expressions
    score := 85
    grade := score >= 90 ? "A" : score >= 80 ? "B" : score >= 60 ? "C" : "D"
    fmt.Println(grade) // Output: B

    // Numeric types
    max := 10 > 5 ? 10 : 5
    fmt.Println(max) // Output: 10
}
```

### Elegant usage with Optional Chaining

MyGO supports the Elvis operator `x ?: y`. It allows null-coalescing operations and is a special case of the ternary operator.

Since optional chaining returns pointer types, they need to be dereferenced. However, dereferencing `nil` directly causes a panic. The following approach is recommended:

```go
func main() {
    user1 := &User{Name: "Alice", Profile: &Profile{Email: "alice@example.com", Age: 25}}
    user2 := &User{Name: "Bob", Profile: nil}

    // Traditional check
    if user2.Profile != nil {
        fmt.Println(user2.Profile.Email)
    }

    // Ternary + Optional Chaining
    email1 := (user1?.Profile?.Email != nil) ? *user1?.Profile?.Email : ""
    email2 := (user2?.Profile?.Email != nil) ? *user2?.Profile?.Email : ""

    fmt.Println("email1:", email1) // alice@example.com
    fmt.Println("email2:", email2) // ""

    // New syntax: Elvis Operator
    // x ?: y is equivalent to (x != nil) ? *x : y
    email1 := user1?.Profile?.Email ?: ""
    email2 := user2?.Profile?.Email ?: ""

    fmt.Println("email1:", email1) // alice@example.com
    fmt.Println("email2:", email2) // ""
}
```

## 5. Struct Method Overloading

Supports defining multiple methods with the same name for a struct, as long as parameter types differ. The compiler automatically selects the correct method based on call arguments.

### 5.1 Basic Usage

```go
type Calculator struct{}

// Integer addition
func (c *Calculator) Add(a int, b int) int {
    return a + b
}

// Float addition
func (c *Calculator) Add(a float64, b float64) float64 {
    return a + b
}

// String concatenation
func (c *Calculator) Add(a string, b string) string {
    return a + b
}

func main() {
    calc := &Calculator{}

    intResult := calc.Add(1, 2)             // Calls int version
    floatResult := calc.Add(1.5, 2.5)       // Calls float64 version
    strResult := calc.Add("Hello", "GO")    // Calls string version

    fmt.Printf("int: %T = %v\n", intResult, intResult)         // int: int = 3
    fmt.Printf("float64: %T = %v\n", floatResult, floatResult) // float64: float64 = 4.0
    fmt.Printf("string: %T = %v\n", strResult, strResult)      // string: string = HelloGO
}
```

### 5.2 Different Parameter Counts

```go
type Greeter struct{}

// No args
func (g *Greeter) SayHello() string {
    return "Hello, World!"
}

// One arg
func (g *Greeter) SayHello(name string) string {
    return "Hello, " + name + "!"
}

// Two args
func (g *Greeter) SayHello(firstName string, lastName string) string {
    return "Hello, " + firstName + " " + lastName + "!"
}

func main() {
    g := &Greeter{}

    fmt.Println(g.SayHello())                       // Output: Hello, World!
    fmt.Println(g.SayHello("Alice"))                // Output: Hello, Alice!
    fmt.Println(g.SayHello("Bob", "Smith"))         // Output: Hello, Bob Smith!
}
```

### 5.3 Different Return Types

```go
type Processor struct{}

// Returns squared int
func (p *Processor) Process(x int) int {
    return x * x
}

// Returns sqrt float
func (p *Processor) Process(x float64) float64 {
    return math.Sqrt(x)
}

// Returns string length
func (p *Processor) Process(s string) int {
    return len(s)
}

func main() {
    proc := &Processor{}

    intResult := proc.Process(5)       // int: 25
    floatResult := proc.Process(16.0)  // float64: 4.0
    lenResult := proc.Process("MyGO")  // int: 4

    // No type assertion needed! Compiler knows exact return type
    fmt.Printf("%T: %v\n", intResult, intResult)
    fmt.Printf("%T: %v\n", floatResult, floatResult)
    fmt.Printf("%T: %v\n", lenResult, lenResult)
}
```

### 5.4 Overload Resolution

When multiple overload candidates exist, MyGO selects the target based on "More Specific Priority":

1.  **Fixed-length parameters (non-variadic) preferred**
    *   If a "Strong Match" fixed-length overload exists, `...` versions are ignored.
    *   `...` (variadic) acts only as a **fallback**.

2.  **Same Type / More Specific Type Match preferred**
    *   **Exact Match** (e.g., `int` arg matches `int` param) has highest priority.
    *   **Same Kind Strong Match** (e.g., int literal matches various `int/uint` params) is next.
    *   **Weak Match** (e.g., `int` literal matching `float32/float64`) happens only if no better candidate exists.
    *   `any/interface{}` and `unknown` (expressions whose type cannot be inferred before compilation) are weaker matches.

3.  **Variadic can win over fixed-length if fixed-length only matches via "unknown/any"**
    This prevents selecting a generic fixed-length overload when a specific variadic one exists but type info is incomplete during pre-typecheck.

> Note: Overload selection happens during pre-typecheck. If types are `unknown`, the compiler makes conservative inferences but tries to let the "more precise" candidate win.

```go
type NDArray struct{ shape []int }

func (a *NDArray) _init(shape ...int) {}
func (a *NDArray) _init(data []float64, shape []int) {}

func f(a *NDArray) {
    // rows/cols inferred as int, otherwise treated as unknown
    rows, cols := a.shape[0], a.shape[1]
    _ = make(NDArray, rows, cols) // Expects _init(...int)
}
```

### 5.5 Practical Example

```go
type DataStore struct {
    intData    map[string]int
    stringData map[string]string
}

// Store Int
func (ds *DataStore) Set(key string, value int) {
    if ds.intData == nil {
        ds.intData = make(map[string]int)
    }
    ds.intData[key] = value
}

// Store String
func (ds *DataStore) Set(key string, value string) {
    if ds.stringData == nil {
        ds.stringData = make(map[string]string)
    }
    ds.stringData[key] = value
}

// Get Int
func (ds *DataStore) Get(key string, defaultValue int) int {
    if v, ok := ds.intData[key]; ok {
        return v
    }
    return defaultValue
}

// Get String
func (ds *DataStore) Get(key string, defaultValue string) string {
    if v, ok := ds.stringData[key]; ok {
        return v
    }
    return defaultValue
}

func main() {
    store := &DataStore{}

    store.Set("age", 25)
    store.Set("name", "Alice")

    // Type-safe retrieval
    age := store.Get("age", 0)           // int
    name := store.Get("name", "Unknown") // string

    fmt.Printf("Age: %d, Name: %s\n", age, name) // Age: 25, Name: Alice
}
```

## 6. Magic Functions (Experimental)

Note: Experimental features may have bugs. Use with caution. (All magic methods are primarily designed for future vector calculations).

### Constructors

Use `make(TypeName, args...)` syntax to create struct instances with custom initialization logic.

**Rules**:
*   Method must be named `_init`.
*   Must be a pointer receiver (`func (t *Type) _init(...)`).
*   No return value needed (compiler adds it automatically).
*   Supports overloading and default arguments.

#### Basic Usage

```go
type Person struct {
    name string
    age  int
}

func (p *Person) _init(name string, age int) {
    p.name = name
    p.age = age
}

func main() {
    p := make(Person, "Alice", 25)
    fmt.Println(p.name, p.age) // Output: Alice 25
}
```

#### Constructor Overloading

```go
type Database struct {
    host     string
    port     int
    username string
}

// Init via port
func (d *Database) _init(host string, port int) {
    d.host = host
    d.port = port
    d.username = "admin"
}

// Init via username
func (d *Database) _init(host string, username string) {
    d.host = host
    d.port = 3306 // Default port
    d.username = username
}

func main() {
    db1 := make(Database, "localhost", 3306)   // Calls first _init
    db2 := make(Database, "localhost", "root") // Calls second _init

    fmt.Println(db1.host, db1.port, db1.username) // localhost 3306 admin
    fmt.Println(db2.host, db2.port, db2.username) // localhost 3306 root
}
```

#### Constructor + Default Arguments

```go
type Server struct {
    host string
    port int
}

func (s *Server) _init(host string, port int = 8080) {
    s.host = host
    s.port = port
}

func main() {
    s1 := make(Server, "localhost")       // Use default port 8080
    s2 := make(Server, "0.0.0.0", 3000)   // Specify port

    fmt.Printf("%s:%d\n", s1.host, s1.port) // localhost:8080
    fmt.Printf("%s:%d\n", s2.host, s2.port) // 0.0.0.0:3000
}
```

### Index Operator Overloading (_getitem / _setitem)

MyGo supports custom indexing via `_getitem` and `_setitem`.

#### Rules

1.  **Comma present** → Forces matching `[]T` (slice) params.
2.  **No comma** → Prefers `T` param, falls back to `[]T`.

> `_getitem/_setitem` follows the **Unified Priority** (fixed-length > same type > weak match) after applying the "Comma/No-Comma" filter.

#### Basic Example

```go
package main

import "fmt"

type Matrix struct {
    data [][]int
    rows int
    cols int
}

func (m *Matrix) _init(rows, cols int) {
    m.rows = rows
    m.cols = cols
    m.data = make([][]int, rows)
    for i := range m.data {
        m.data[i] = make([]int, cols)
    }
}

// _getitem: Supports matrix[row, col] syntax
func (m *Matrix) _getitem(indices1 []int, indices2 []int) int {
    row, col := indices1[0], indices2[0]
    return m.data[row][col]
}

// _setitem: Supports matrix[row, col] = value
// ⚠️ Note: value comes FIRST, indices follow
func (m *Matrix) _setitem(value int, indices1 []int, indices2 []int) {
    row, col := indices1[0], indices2[0]
    m.data[row][col] = value
}

func main() {
    m := make(Matrix, 3, 3)

    // Set value - calls _setitem
    m[0, 0] = 1
    m[1, 1] = 5
    m[2, 2] = 9

    // Get value - calls _getitem
    fmt.Println(m[0, 0]) // Output: 1
    fmt.Println(m[1, 1]) // Output: 5
    fmt.Println(m[2, 2]) // Output: 9
}
```

#### Comma Syntax vs. Colon Syntax

MyGO distinguishes between Comma Separated and Colon Slice syntax:

```go
package main

import "fmt"

type NDArray struct {
    data []int
}

func (a *NDArray) _init(data []int) {
    a.data = data
}

// Colon syntax: arr[start:end] -> Passes normal int args
func (a *NDArray) _getitem(args ...int) []int {
    fmt.Printf("Slice access: %v\n", args)
    start, end := args[0], args[1]
    return a.data[start:end]
}

// Comma syntax: arr[i, j, k] -> Passes slice args
func (a *NDArray) _getitem(indices ...[]int) int {
    fmt.Printf("Multi-dim index: %v\n", indices)
    return 0
}

func main() {
    arr := make(NDArray, []int{1, 2, 3, 4, 5})

    // Colon syntax - matches ...int
    _ = arr[1:3] // Output: Slice access: [1, 3]

    // Comma syntax - matches ...[]int
    _ = arr[1, 2]       // Output: Multi-dim index: [[1], [2]]
    _ = arr[1:2, 3:4]   // Output: Multi-dim index: [[1, 2], [3, 4]]
}
```

#### Usage Case

```go
package main

import "fmt"

type Person struct {
    data map[string]string
}

func (p *Person) _init() {
    p.data = make(map[string]string)
    p.data["name"] = "Alice"
    p.data["age"] = "25"
    p.data["city"] = "Beijing"
}

// _getitem: Supports person["name"]
func (p *Person) _getitem(name string) string {
    if value, ok := p.data[name]; ok {
        return value
    }
    return "not found"
}

// _setitem: Supports person["name"] = value
func (p *Person) _setitem(value string, name string) {
    p.data[name] = value
}

func main() {
    person := make(Person)

    fmt.Println("Name:", person["name"]) // Output: Name: Alice
    fmt.Println("Age:", person["age"])   // Output: Age: 25

    person["name"] = "Bob"
    person["country"] = "China"

    fmt.Println("Updated Name:", person["name"]) // Output: Updated Name: Bob
    fmt.Println("Country:", person["country"])   // Output: Country: China
    fmt.Println("Unknown:", person["unknown"])   // Output: Unknown: not found
}
```

### Arithmetic Operator Overloading

MyGO supports overloading standard arithmetic operators (`+`, `-`, `*`, `/`, etc.) via specific methods named `_add`, `_sub`, etc.

**Resolution Rules**

The compiler parses binary operations (e.g., `a + b`) in this order:

1.  **Forward Priority**: Try left operand's method (`a._add(b)`).
2.  **Reverse Fallback**: If not found/matched, try right operand's reverse method (`b._radd(a)`).
3.  **Unary Special**: Increment (`++`) and Decrement (`--`) only support forward calls.

##### Binary Operator Table

| Operator | Forward (Primary) | Reverse (Backup) | Note |
| :---: | :--- | :--- | :--- |
| `+` | `func _add(b T) T` | `func _radd(a T) T` | Add |
| `-` | `func _sub(b T) T` | `func _rsub(a T) T` | Subtract |
| `*` | `func _mul(b T) T` | `func _rmul(a T) T` | Multiply |
| `/` | `func _div(b T) T` | `func _rdiv(a T) T` | Divide |
| `%` | `func _mod(b T) T` | `func _rmod(a T) T` | Modulo |
| `++` | `func _inc()` | N/A | Increment (No return) |
| `--` | `func _dec()` | N/A | Decrement (No return) |

##### Unary Operator Table

| Operator | Forward | Reverse | Note |
| :--: | :----------------- | :--- | :-------------- |
| `+a` | `func _pos() T` | N/A | Unary Plus |
| `-a` | `func _neg() T` | N/A | Unary Negate |
| `^a` | `func _invert() T` | N/A | Bitwise Not |

##### Comparison Operator Table

| Operator | Forward | Mirror Fallback | Note |
| :--: | :------------------- | :----- | :--- |
| `==` | `func _eq(v T) bool` | `_eq` | Equal |
| `!=` | `func _ne(v T) bool` | `_ne` | Not Equal |
| `>` | `func _gt(v T) bool` | `_lt` | Greater |
| `>=` | `func _ge(v T) bool` | `_le` | Greater/Equal |
| `<` | `func _lt(v T) bool` | `_gt` | Less |
| `<=` | `func _le(v T) bool` | `_ge` | Less/Equal |

**Mirror Fallback Rule (Important)**

For expression `a OP b`:
If `a` does not implement the method, the compiler tries the **Mirror Method** of the right operand `b`.

| Expression | Fallback |
| -------- | ---------- |
| `a < b` | `b._gt(a)` |
| `a <= b` | `b._ge(a)` |
| `a > b` | `b._lt(a)` |
| `a >= b` | `b._le(a)` |
| `a == b` | `b._eq(a)` |
| `a != b` | `b._ne(a)` |

**⚠️ Pointer Warning**
If a pointer type overloads `==` or `!=`, it overrides native pointer address comparison. The compiler issues a warning:
`warning: *T defines _eq (== overload) or _ne (!= overload), which overrides the native pointer == or != semantics for this type`

##### Bitwise Operator Table

| Operator | Forward | Reverse | Note |
| :--: | :---------------------- | :----------------------- | :--- |
| `\|` | `func _or(v T) T` | `func _ror(v T) T` | OR |
| `&` | `func _and(v T) T` | `func _rand(v T) T` | AND |
| `^` | `func _xor(v T) T` | `func _rxor(v T) T` | XOR |
| `<<` | `func _lshift(v T) T` | `func _rlshift(v T) T` | L-Shift |
| `>>` | `func _rshift(v T) T` | `func _rrshift(v T) T` | R-Shift |
| `&^` | `func _bitclear(v T) T` | `func _rbitclear(v T) T` | Bit Clear |

##### Data Flow Operators

| Operator | Position | Method | Reverse | Note |
| :------: | :-: | :---------------- | :--- | :- |
| `<-a` | Prefix | `func _recv() T` | N/A | Receive |
| `a <- v` | Infix | `func _send(v T)` | N/A | Send |

**⚠️ About `select`**

At present, `select` does not directly support `_recv() T` and `_send(v T)`.
However, you can enable `select` support for custom structs by overloading the `Chan() chan` method.

That said, this approach may introduce **ambiguity**, so it is **not recommended**.

##### Compound Assignment Expansion

If a type implements the operator, compound assignments expand automatically:

| Syntax | Expansion |
| --------- | ------------ |
| `a += b` | `a = a + b` |
| `a -= b` | `a = a - b` |
| `a &= b` | `a = a & b` |
| `a &^= b` | `a = a &^ b` |
| `a <<= b` | `a = a << b` |

#### Vector Math Example

```go
package main

import "fmt"

// Vector supports forward + and ++
type Vector struct {
    x, y int
}

// _add: a + b
func (v *Vector) _add(other *Vector) *Vector {
    return &Vector{x: v.x + other.x, y: v.y + other.y}
}

// _sub: a - b
func (v *Vector) _sub(other *Vector) *Vector {
    return &Vector{x: v.x - other.x, y: v.y - other.y}
}

// _inc: v++
func (v *Vector) _inc() {
    v.x++
    v.y++
}

func main() {
    v1 := &Vector{x: 1, y: 1}
    v2 := &Vector{x: 2, y: 3}

    // Basic math
    v3 := v1 + v2 // Calls v1._add(v2)
    fmt.Println(v3) // Output: &{3 4}

    // Chained
    v4 := v1 + v2 - v1 // (v1._add(v2))._sub(v1)
    fmt.Println(v4) // Output: &{2 3}

    // Increment
    v1++ // Calls v1._inc()
    fmt.Println(v1) // Output: &{2 2}
}
```

#### Mixed Types and Reverse Operations

MyGo uses Reverse Operators to handle mixed-type math (e.g., `Vector` + `Scalar`). If `Vector` doesn't define `_add(Scalar)`, `Scalar` can define `_radd(Vector)`.

```go
package main

import "fmt"

type NDArray struct {
    data []int
}

// _add: NDArray + NDArray
func (a *NDArray) _add(b *NDArray) *NDArray {
    res := make([]int, len(a.data))
    for i, v := range a.data {
        res[i] = v + b.data[i]
    }
    return &NDArray{data: res}
}

type Scalar struct {
    val int
}

// _radd: Handles NDArray + Scalar
// Called if left (NDArray) _add(Scalar) is missing
func (s *Scalar) _radd(arr *NDArray) *NDArray {
    fmt.Println("Trigger reverse op: Scalar._radd")
    res := make([]int, len(arr.data))
    for i, v := range arr.data {
        res[i] = v + s.val
    }
    return &NDArray{data: res}
}

func main() {
    arr := &NDArray{data: []int{10, 20, 30}}
    num := &Scalar{val: 5}

    // 1. Same type
    sumArr := arr + arr
    fmt.Println("Vec + Vec:", sumArr.data)
    // Output: [20 40 60]

    // 2. Mixed type (Reverse trigger)
    // a. arr._add(num) -> Not found
    // b. num._radd(arr) -> Found & Called
    mixed := arr + num
    fmt.Println("Vec + Scalar:", mixed.data)
    // Output: Trigger reverse op: Scalar._radd
    // Output: [15 25 35]
}
```

### Generics Support

MyGO extends operator overloading to Generics. You can use `+`, `-`, `==`, `[]` etc., on generic type `T` if `T` satisfies a constraint interface with the "magic methods".

**Mechanism**

*   Constraint: `T`'s interface constraint must declare the magic method (e.g., `_add`).
*   Rewrite: Compiler verifies `T` meets constraint, then rewrites `a + b` to `a._add(b)`.

#### Example Code

```go
package main

import "fmt"

// 1. Define constraint
type Addable[T any] interface {
    _add(T) T
}

// 2. Custom Struct
type MyInt struct {
    Val int
}

func (m MyInt) _add(other MyInt) MyInt {
    return MyInt{Val: m.Val + other.Val}
}

// 3. Generic Function
// T must be Addable[T]
func GenericAdd[T Addable[T]](a, b T) T {
    // Compiler allows + because of _add constraint
    // Rewrites to a._add(b)
    return a + b
}

func main() {
    // A. Custom Type
    v1 := MyInt{10}
    v2 := MyInt{20}
    sumObj := GenericAdd(v1, v2)
    fmt.Println(sumObj) // Output: {30}

    // B. Native Types (MyGO Feature)
    // Compiler synthesizes _add for int/float/etc.
    sumInt := GenericAdd(100, 200)
    fmt.Println(sumInt) // Output: 300
}
```

##### Generic Constructors

MyGO supports constructors for Generic types. `make(GenericType[T], ...)` matches the specific `T`'s `_init`.

```go
package main

import "fmt"

type Box[T any] struct {
    Value T
    Tag   string
}

// Generic Constructor
func (b *Box[T]) _init(val T, tag string) {
    b.Value = val
    b.Tag = tag
}

// Overload for Generic
func (b *Box[T]) _init() {
    b.Tag = "default"
}

func main() {
    // 1. Instantiate int
    b1 := make(Box[int], 100, "manual")

    // 2. Instantiate string
    b2 := make(Box[string])

    fmt.Printf("b1: %v, %s\n", b1.Value, b1.Tag) // b1: 100, manual
    fmt.Printf("b2: %q, %s\n", b2.Value, b2.Tag) // b2: "", default
}
```

#### Native Types Support

MyGO compiler synthesizes methods for native types (`int`, `float64`, `string`, `slice`, `map`). They are **Zero-Cost Abstractions** (compiled to native IR, no boxing).

Case:

```go
// Slice automatically satisfies _getitem constraint
func GetFirst[T any, S interface{ _getitem(int) T }](seq S) T {
    return seq[0] // seq._getitem(0)
}

func main() {
    list := []int{1, 2, 3}
    println(GetFirst(list)) // Output: 1
}
```

**Native Type Method Synthesis Table**

| Native Type                                                | Synthesized Methods                          | Semantics / Lowering (Conceptual)                                              |
| ---------------------------------------------------------- | -------------------------------------------- | ------------------------------------------------------------------------------ |
| Numeric basic (`int` / `uint` / `float` / `complex`, etc.) | `_add / _sub / _mul / _div / _mod`           | `a + b` / `a - b` / `a * b` / `a / b` / `a % b`                                |
| Numeric basic                                              | `_radd / _rsub / _rmul / _rdiv / _rmod`      | Reverse operation: `b OP a` (used when operand swapping is required)           |
| Numeric basic                                              | `_and / _or / _xor / _bitclear`              | `a & b` / `a \| b` / `a ^ b` / `a &^ b`                                        |
| Numeric basic                                              | `_rand / _ror / _rxor / _rbitclear`          | Reverse bitwise operation: `b OP a`                                            |
| Numeric basic (integers only)                              | `_lshift / _rshift`                          | `a << b` / `a >> b`                                                            |
| Numeric basic (integers only)                              | `_rlshift / _rrshift`                        | Reverse shift: `b << a` / `b >> a`                                             |
| Numeric basic / `string`                                   | `_eq / _ne / _lt / _le / _gt / _ge`          | `a == b` / `!=` / `<` / `<=` / `>` / `>=`                                      |
| Numeric basic                                              | `_pos / _neg / _invert`                      | `+a` / `-a` / `^a`                                                             |
| `string`                                                   | `_add / _radd`                               | String concatenation: `a + b` (including operand swap)                         |
| `slice`                                                    | `_getitem(int) T` / `_setitem(int, T)`       | `seq[i]` / `seq[i] = v`                                                        |
| `map`                                                      | `_getitem(K) V` / `_setitem(V, K)`           | `m[k]` / `m[k] = v`                                                            |
| `chan`                                                     | `_send(T)` / `_recv() T`                     | Data flow: `ch <- v` / `<-ch`                                                  |
| `slice`                                                    | `_init(len int)` / `_init(len int, cap int)` | `make([]T, len)` / `make([]T, len, cap)`                                       |
| `map` / `chan`                                             | `_init()` / `_init(size int)`                | `make(map[K]V)` / `make(map[K]V, size)`; `make(chan T)` / `make(chan T, size)` |


**⚠️ Warning ⚠️**
`type Name float64` declarations can be overloaded, but it is **not recommended**. It causes boxing of the primitive type, hurting performance, and the type is no longer treated as its underlying type.

## 7. Algebraic Data Types (Experimental Feature)

### Enums

MyGO supports `enum` (Tagged Union / ADT). An enum type consists of multiple **`Variants`**, where each variant can carry a payload of a different type.

PS: There were quite a few bugs during the initial testing of this feature. I have tested all the boundary conditions I could think of, but I am not sure if other bugs remain.

#### 7.1 Defining Enums

```go
// Unit variants (no payload)
type Color enum {
	Red
	Green
	Blue
}

// Payload variants (with payload, supports multi-parameter tuples)
type Shape enum {
	Circle(float64)
	Rect(float64, float64)
	Point
}


// Recursive enums are also supported
type List enum {
	Cons(int, List)
	Nil
}

// Generic enums
type Option[T any] enum {
	Some(T)
	None
}
```

#### 7.2 Constructing Enum Values

```go
c1 := Color.Red            // unit variant: no parentheses needed
s1 := Shape.Circle(1.5)    // payload variant: requires arguments
s2 := Shape.Rect(3, 4)
s3 := Shape.Point

o1 := Option[int].Some(42)
o2 := Option[int].None
```

#### 7.3 Pattern Matching

MyGO allows conditional branching based on the data structure of an `enum` at compile time.

```go
func area(s Shape) float64 {
	switch s {
	case Shape.Circle(r):
		return 3.14159 * r * r
	case Shape.Rect(w, h):
		return w * h
	case Shape.Point:
		return 0
	default:
		return 0
	}
}
```

**Wildcard**: Use `_` to ignore fields you don't care about.

```go
switch s {
case Shape.Rect(_, _):
	// any rectangle
}
```

##### Exhaustiveness Checking

MyGO's `enum` supports **compile-time exhaustiveness checking** in `pattern matching`. This ensures that all possible `Variants` are explicitly handled, avoiding logical errors caused by missing branches.

If a `switch` is missing certain `Variants` and has no `default` branch, the compiler will report an error:

```go
package main

import "fmt"

type Shape enum {
	Circle(float64)
	Rect(float64, float64)
	Point
}

func area(s Shape) float64 {
	switch s {
	case Shape.Circle(r):
		return 3.14159 * r * r
	case Shape.Rect(w, h):
		return w * h
	}
}

func main() {
	s := Shape.Circle(1.5)
	fmt.Println(area(s))
}

// Error: enum match on Shape is not exhaustive (missing: Point)
```

#### 7.4 Generics + Shape (GC Shape) and Storage Strategy (stack/heap)

Enums are lowered into standard Go `structs` + type constructors (using `unsafe` internally). To reduce GC pressure:
- **Pure value payloads** (no pointers) prefer `_stack [N]byte` (stack inlining).
- **Pointer-containing payloads** use `_heap unsafe.Pointer`.
- **Generic enums**: Determined by the *instantiated concrete type shape* (e.g., `Option[int]` vs `Option[string]`).

In generic functions (e.g., `func f[T any](o Option[T])`), if the shape is unknown at compile time, pattern matching/reading payloads will use runtime branching to select between `_heap` and `_stack`.

#### 7.5 if/for Pattern Matching

In addition to `switch` pattern matching, MyGO supports cleaner syntax for `if`/`for` pattern matching. (Note: These do not have exhaustiveness checking).

##### if Pattern Matching

```go
opt := Option[int].Some(42)

// Basic form
if Option.Some(x) := opt {
    fmt.Println("value:", x)
}

// With else
if Option.Some(x) := opt {
    fmt.Println("value:", x)
} else {
    fmt.Println("no value")
}

// With guard
if Option.Some(x) := opt; x > 0 {
    fmt.Println("positive:", x)
}


// else-if chain
if Option.Some(x) := opt {
	fmt.Println("opt1:", x)
} else if Option.Some(y) := opt {
	fmt.Println("opt2:", y)
} else {
	fmt.Println("both none")
}
```

##### for Pattern Matching

The `for` pattern match loop continues executing until the pattern no longer matches:

```go
// Basic form: loops until mismatch
shape := Shape.Circle(1.5)
for Shape.Circle(r) := shape {
    fmt.Println("radius:", r)
    shape = Shape.Point  // Change value to exit loop
}

// With guard
shape2 := Shape.Circle(5.0)
for Shape.Circle(r) := shape2; r > 1.0 {
    fmt.Println("r =", r)
    shape2 = Shape.Circle(r - 2.0)  // Decrement, exit when r <= 1.0
}

// Multi-field destructuring
rect := Shape.Rect(3.0, 4.0)
for Shape.Rect(w, h) := rect {
    fmt.Println("area:", w*h)
    rect = Shape.Point
}
```

#### 7.6 "Magic Functions" support for Enums (Operator Dispatchers)

You can define magic methods (other than `_init`) for enums, such as `_add/_radd/_eq/_ne/_lt/_getitem/...`, letting the enum itself act as a **dynamic dispatcher**:

```go
type Value enum {
	Integer(int)
	Float(float64)
}

// Supports Value + Value (homogenous/heterogenous dispatch here)
func (lhs Value) _add(rhs Value) Value {
	switch lhs {
	case Value.Integer(a):
		switch rhs {
		case Value.Integer(b):
			return Value.Integer(a + b)
		case Value.Float(b):
			return Value.Float(float64(a) + b)
		}
	case Value.Float(a):
		switch rhs {
		case Value.Integer(b):
			return Value.Float(a + float64(b))
		case Value.Float(b):
			return Value.Float(a + b)
		}
	}
	panic("unsupported Value + Value")
}

func main() {
	x := Value.Integer(1)
	y := Value.Float(2.5)
	z := x + y // Rewritten as x._add(y)
	switch z {
	case Value.Integer(a):
		fmt.Printf("z := %d\n", a)
	case Value.Float(b):
		fmt.Printf("z := %f\n", b)
	}
}
```

**Reverse Operation**: If the forward `_add` does not match, it attempts the right-side `_radd` (consistent with the magic function rules in the README).

**Note**:
- Writing `a + b` inside the magic method might trigger recursive rewriting; it is recommended to write dispatch logic directly inside the magic method body and avoid calling itself with the same operator.

#### 7.7 Enum Usage Examples

##### Constructing Option[T]

`Option[T]` is used to explicitly represent "Present / Absent", avoiding nil, multiple return values, and implicit errors.

```go
type Divable interface {
	~int | ~int8 | ~int16 | ~int32 | ~int64 |
		~uint | ~uint8 | ~uint16 | ~uint32 | ~uint64 | ~uintptr |
		~float32 | ~float64
}

type Option[T Divable] enum {
	Some(T)
	None
}

func (o Option[T]) _eq(a Option[T]) bool {
	switch o {
	case Option[T].Some(_):
		switch a {
		case Option[T].Some(_):
			return true
		case Option[T].None:
			return false
		}
	case Option[T].None:
		switch a {
		case Option[T].None:
			return true
		case Option[T].Some(_):
			return false
		}
	}
	panic("unreachable")
}

func (o Option[T]) _div(a Option[T]) Option[T] {
	switch o {
	case Option[T].Some(v):
		switch a {
		case Option[T].None:
			return Option[T].None
		case Option[T].Some(d):
			var zero T
			if d == zero {
				return Option[T].None
			}
			return Option[T].Some(v / d)
		}
	case Option[T].None:
		return Option[T].None
	}
	panic("unreachable")
}

func main() {
	x := Option[int].Some(10) / Option[int].Some(2)
	y := Option[int].Some(10) / Option[int].Some(0)

	switch x {
	case Option[int].Some(v):
		fmt.Println("x =", v)
	case Option[int].None:
		fmt.Println("x = None")
	}

	switch y {
	case Option[int].Some(v):
		fmt.Println("y =", v)
	case Option[int].None:
		fmt.Println("y = None")
	}
}
```

##### Constructing Result[T, E]

`Result[T, E]` is used for explicit error propagation, making it more suitable for expressing structured failure reasons than `error`.

```go
type Result[T any, E error] enum {
	Ok(T)
	Err(E)
}

// Example 1: Safe division, returning Result[int, error]
func safeDiv(a, b int) Result[int, error] {
	if b == 0 {
		return Result[int, error].Err(errors.New("division by zero"))
	}
	return Result[int, error].Ok(a / b)
}

// Example 2: String to int, returning Result[int, error]
func parseIntSafe(s string) Result[int, error] {
	val, err := strconv.Atoi(s)
	if err != nil {
		return Result[int, error].Err(err)
	}
	return Result[int, error].Ok(val)
}

// Example 3: Map operation on Result (transforming the value inside Ok)
func (r Result[T, E]) mapOk(f func(T) T) Result[T, E] {
	switch r {
	case Result[T, E].Ok(val):
		return Result[T, E].Ok(f(val))
	case Result[T, E].Err(e):
		return Result[T, E].Err(e)
	}
	panic("unreachable")
}

func main() {
	// Test safe division
	fmt.Println("=== Safe Division Example ===")
	r1 := safeDiv(10, 2)
	switch r1 {
	case Result[int, error].Ok(v):
		fmt.Printf("10 / 2 = %d\n", v)
	case Result[int, error].Err(e):
		fmt.Printf("Error: %v\n", e)
	}

	r2 := safeDiv(10, 0)
	switch r2 {
	case Result[int, error].Ok(v):
		fmt.Printf("10 / 0 = %d\n", v)
	case Result[int, error].Err(e):
		fmt.Printf("Error: %v\n", e)
	}

	// Test string parsing
	fmt.Println("\n=== String Parsing Example ===")
	r3 := parseIntSafe("42")
	switch r3 {
	case Result[int, error].Ok(v):
		fmt.Printf("Parse success: %d\n", v)
	case Result[int, error].Err(e):
		fmt.Printf("Parse failure: %v\n", e)
	}

	r4 := parseIntSafe("not a number")
	switch r4 {
	case Result[int, error].Ok(v):
		fmt.Printf("Parse success: %d\n", v)
	case Result[int, error].Err(e):
		fmt.Printf("Parse failure: %v\n", e)
	}

	// Test map operation
	fmt.Println("\n=== Result map Operation Example ===")
	r5 := parseIntSafe("5").mapOk(func(x int) int {
		return x * 2
	})
	switch r5 {
	case Result[int, error].Ok(v):
		fmt.Printf("5 * 2 = %d\n", v)
	case Result[int, error].Err(e):
		fmt.Printf("Error: %v\n", e)
	}

	r6 := parseIntSafe("invalid").mapOk(func(x int) int {
		return x * 2
	})
	switch r6 {
	case Result[int, error].Ok(v):
		fmt.Printf("Result: %d\n", v)
	case Result[int, error].Err(e):
		fmt.Printf("Error: %v\n", e)
	}
}
```

##### Monad Style

MyGO's `enum` + `magic functions` naturally support `Monad` / `Functor` patterns.

###### Map / AndThen / UnwrapOr

```go
package main

import (
	"errors"
	"fmt"
	"strconv"
)

type Divable interface {
	~int | ~int8 | ~int16 | ~int32 | ~int64 |
		~uint | ~uint8 | ~uint16 | ~uint32 | ~uint64 | ~uintptr |
		~float32 | ~float64
}

type Option[T Divable] enum {
	Some(T)
	None
}

func (o Option[T]) _eq(a Option[T]) bool {
	switch o {
	case Option[T].Some(_):
		switch a {
		case Option[T].Some(_):
			return true
		case Option[T].None:
			return false
		}
	case Option[T].None:
		switch a {
		case Option[T].None:
			return true
		case Option[T].Some(_):
			return false
		}
	}
	panic("unreachable")
}

// Monad Style API (Option)
func (o Option[T]) Map(f func(T) T) Option[T] {
	switch o {
	case Option[T].Some(v):
		return Option[T].Some(f(v))
	case Option[T].None:
		return Option[T].None
	}
	panic("unreachable")
}

func (o Option[T]) AndThen(f func(T) Option[T]) Option[T] {
	switch o {
	case Option[T].Some(v):
		return f(v)
	case Option[T].None:
		return Option[T].None
	}
	panic("unreachable")
}

func (o Option[T]) UnwrapOr(def T) T {
	switch o {
	case Option[T].Some(v):
		return v
	case Option[T].None:
		return def
	}
	panic("unreachable")
}

func (o Option[T]) _div(a Option[T]) Option[T] {
	switch o {
	case Option[T].Some(v):
		switch a {
		case Option[T].None:
			return Option[T].None
		case Option[T].Some(d):
			var zero T
			if d == zero {
				return Option[T].None
			}
			return Option[T].Some(v / d)
		}
	case Option[T].None:
		return Option[T].None
	}
	panic("unreachable")
}

type Result[T any, E error] enum {
	Ok(T)
	Err(E)
}

// Monad Style API (Result)
func (r Result[T, E]) Map(f func(T) T) Result[T, E] {
	switch r {
	case Result[T, E].Ok(v):
		return Result[T, E].Ok(f(v))
	case Result[T, E].Err(e):
		return Result[T, E].Err(e)
	}
	panic("unreachable")
}

func (r Result[T, E]) AndThen(f func(T) Result[T, E]) Result[T, E] {
	switch r {
	case Result[T, E].Ok(v):
		return f(v)
	case Result[T, E].Err(e):
		return Result[T, E].Err(e)
	}
	panic("unreachable")
}

// unwrapOr: returns default value on error (no panic)
func (r Result[T, E]) UnwrapOr(def T) T {
	switch r {
	case Result[T, E].Ok(v):
		return v
	case Result[T, E].Err(_):
		return def
	}
	panic("unreachable")
}

// unwrapOrHandle: calls onErr on error then returns default value (used for "unwrapOr with error handling")
func (r Result[T, E]) UnwrapOrHandle(def T, onErr func(E)) T {
	switch r {
	case Result[T, E].Ok(v):
		return v
	case Result[T, E].Err(e):
		if onErr != nil {
			onErr(e)
		}
		return def
	}
	panic("unreachable")
}

// unwrap: panics on error (closer to Rust's unwrap)
func (r Result[T, E]) Unwrap() T {
	switch r {
	case Result[T, E].Ok(v):
		return v
	case Result[T, E].Err(e):
		panic(e)
	}
	panic("unreachable")
}

func parseInt(s string) Result[int, error] {
	v, err := strconv.Atoi(s)
	if err != nil {
		return Result[int, error].Err(err)
	}
	return Result[int, error].Ok(v)
}

func safeDiv(a, b int) Result[int, error] {
	if b == 0 {
		return Result[int, error].Err(errors.New("division by zero"))
	}
	return Result[int, error].Ok(a / b)
}

func printOption(s string, o Option[int]) {
	switch o {
	case Option[int].Some(v):
		fmt.Println(s, "=", v)
	case Option[int].None:
		fmt.Println(s, "= None")
	}
}

func main() {
	// --- Option: / operator + monad chaining ---
	x := Option[int].Some(10) / Option[int].Some(2) // 5
	y := Option[int].Some(10) / Option[int].Some(0) // None

	// map / andThen / unwrapOr
	x2 := x.
		Map(func(v int) int { return v * 2 }).
		AndThen(func(v int) Option[int] { return Option[int].Some(v + 1) })
	fmt.Println("x2 =", x2.UnwrapOr(-1)) // 11
	fmt.Println("y2 =", y.Map(func(v int) int { return v * 2 }).UnwrapOr(-1))

	printOption("x", x)
	printOption("y", y)

	// --- Result: map / andThen / unwrapOr (with error handling) ---
	r1 := parseInt("5").
		Map(func(v int) int { return v * 2 }).
		AndThen(func(v int) Result[int, error] { return safeDiv(100, v) })
	fmt.Println("r1 =", r1.UnwrapOr(-1))

	r2 := parseInt("not a number").
		AndThen(func(v int) Result[int, error] { return safeDiv(100, v) })
	fmt.Println("r2 =", r2.UnwrapOrHandle(-1, func(e error) {
		fmt.Println("r2 err:", e)
	}))
}
```

### Type Algebra

MyGO supports using `+` and `*` in **type expressions** to compose types (structural type algebra).

#### 7.8 Sum Types

`A + B` represents a logical **OR**: at runtime, a value is either `A` or `B`.
It is equivalent to an **anonymous enum**:

```go
// type ID = int + string
// Equivalent (conceptually):
// type ID = enum { int(int); string(string) }
```

* **Variant name (tag / variant name)**:
  By default, it is derived from the operand type’s *printed form* and normalized into an identifier.
  Simple names (e.g. `int`, `error`, `UserStruct`) remain unchanged.
* **Naming of anonymous sum/product types**:
  `A + B` and `B + A` are normalized (sorted) into the same sum type; however, **variant names come from each operand’s printed form**.
  For example, `User * Label` and `Label * User` produce `User_Label` and `Label_User` respectively.
  Therefore, if you need to access concrete variant names (rather than just using the type as a constraint), **give the type an alias**.
* **`nil`**:
  In sum types, `nil` is treated as a **Unit variant** (no payload), used to express Optional types (e.g. `User + nil`).
* **Structural / commutativity**:
  `A + B` and `B + A` are considered the same type (the compiler performs normalization).

---

##### Variant Naming Rules

**Input**: For each operand `Ti` of a sum type, take the *printed form* of its type expression `S = print(Ti)`
(e.g. `int`, `error`, `User * Label`, `pkg.Type[T]`, etc.).

* **Normalization (sanitize)**: Convert `S` into a valid identifier:

  * Allowed characters: letters, digits, underscore (`[A-Za-z0-9_]`)
  * All other characters (spaces, `*`, `[]`, `()`, `.`, `,`, etc.) are collapsed into a single `_`
  * Consecutive `_` are merged
  * Trailing `_` are removed; if the result is empty, use `_`
  * If the first character is a digit, prefix it with `_`
* **`nil` special case**:
  If `Ti == nil`, the variant name is `nil`, and it is a Unit variant (no payload).
* **Conflict resolution**:
  If normalization produces duplicate names (e.g. two different types normalize to the same identifier), `_2`, `_3`, … are appended to ensure uniqueness.

**Examples:**

```go
type ID = int + string
type Result = UserStruct + error
type UserOrNil = UserStruct + nil

type ID = int + string          // Variants: int / string
type R = User * Label + error   // Variants: User_Label / error
type O = User + nil             // Variants: User / nil (nil is Unit)

type Alias = User * Label
type R = Alias + error // Using an alias
```

---

#### 7.9 Product Types

`A * B` represents a logical **AND** (composition). Its behavior depends on the underlying operand types:

* **Interface * Interface**: Interface composition
  (the new type must satisfy both method sets / constraints).
* **Struct * Struct**: Field merging (Mixin).
  The new type contains the combined fields of both structs (order-independent; the compiler normalizes the order).
* **Others (including basic types)**: Not supported for now
  (semantics are intentionally restricted until more general tuple forms are introduced).

---

#### 7.10 Operator Precedence and Disambiguation

Because `*` is used both for pointers and for product types, MyGO uses the following precedence (high → low):

* **Prefix**: `*T`, `[]T` (pointer / slice, etc.) — right-associative
* **Infix**: `A * B` (product types) — left-associative
* **Infix**: `A + B` (sum types) — left-associative

**Examples:**

```go
type T = *User * *Address         // Parsed as: (*User) * (*Address)
type T2 = User * Label + error    // Parsed as: (User * Label) + error
type Complex = *A + B * *C        // Parsed as: (*A) + (B * (*C))
```

---

#### 7.11 Use Cases of Type Algebra

Type algebra can be used to express algebraic constraints abstractly.

```go
package main

import "fmt"

// ============================================
// Layer 1: Interface definitions for basic algebraic structures
// ============================================

// Magma: a closed binary operation
// Requirement: ∀a,b ∈ M, a·b ∈ M
type Magma[T any] interface {
    _mul(T) T
}

// Semigroup: a Magma that satisfies associativity
// Semantic constraint (not checked by the compiler):
// (a·b)·c = a·(b·c)
type Semigroup[T any] interface {
    Magma[T]
}

type Identity[T any] interface {
    _identity() T
}

type Inverse[T any] interface {
    _inverse() T
}

// Monoid: a Semigroup with an identity element
// Semantic constraint: e·a = a·e = a
type Monoid[T any] = Semigroup[T] * Identity[T]

// Group: a Monoid where every element has an inverse
// Semantic constraint: a·a⁻¹ = a⁻¹·a = e
type Group[T any] = Monoid[T] * Inverse[T]

// AbelianGroup: a Group that satisfies commutativity
type AbelianGroup[T any] = Group[T]

// ============================================
// Layer 2: Composing algebraic structures using product types
// ============================================

// Additive structure (Abelian group structure for addition)
type Additive[T any] interface {
    _add(T) T     // a + b
    _neg() T      // -a (additive inverse)
    Zero() T      // 0 (additive identity)
}

// Multiplicative structure (Monoid structure for multiplication)
type Multiplicative[T any] interface {
    _mul(T) T     // a × b
    One() T       // 1 (multiplicative identity)
}

// Multiplicative invertible structure (for non-zero elements of a field)
type MulInvertible[T any] interface {
    Reciprocal() T   // a⁻¹ (multiplicative inverse)
    IsZero() bool    // check whether the value is zero (zero is not invertible)
}

// ============================================
// 🔥 Type algebra definition: Ring = Additive * Multiplicative
// ============================================

// Ring = additive Abelian group * multiplicative monoid
// Semantic constraint: distributive law
// a×(b+c) = a×b + a×c
type Ring[T any] = Additive[T] * Multiplicative[T]

// CommutativeRing: a Ring whose multiplication is commutative
type CommutativeRing[T any] = Ring[T]

// Field = Ring * multiplicative invertibility
// Semantic constraint: non-zero elements form an Abelian group under multiplication
type Field[T any] = Ring[T] * MulInvertible[T]

// ============================================
// Implementation of the integer ring ℤ
// ============================================

type Z int

// --- Additive interface ---
func (a Z) _add(b Z) Z { return a + b }
func (a Z) _neg() Z    { return -a }
func (a Z) Zero() Z    { return 0 }

// --- Multiplicative interface ---
func (a Z) _mul(b Z) Z { return a * b }
func (a Z) One() Z     { return 1 }

// --- Subtraction implemented via addition and negation ---
func (a Z) _sub(b Z) Z { return a + (-b) }

// --- Comparison operations ---
func (a Z) _eq(b Z) bool { return a == b }
func (a Z) _lt(b Z) bool { return a < b }

// ============================================
// Using Ring constraints in generic functions
// ============================================

// Generic power function: works for any Ring
func Pow[T Ring[T]](base T, exp int) T {
    if exp == 0 {
        return base.One()
    }
    result := base.One()
    for i := 0; i < exp; i++ {
        result = result * base  // uses overloaded _mul
    }
    return result
}

// Generic sum: works for any type with an Additive structure
func Sum[T Additive[T]](elements ...T) T {
    if len(elements) == 0 {
        var zero T
        return zero.Zero()
    }
    result := elements[0].Zero()
    for _, e := range elements {
        result = result + e  // uses overloaded _add
    }
    return result
}

// ============================================
// Implementation of the rational field ℚ
// ============================================

type Q struct {
    num int  // numerator
    den int  // denominator
}

// Constructor: automatically reduces the fraction
func (q *Q) _init(num int, den int = 1) {
    if den == 0 {
        panic("denominator cannot be zero")
    }
    // Normalize the sign
    if den < 0 {
        num, den = -num, -den
    }
    // Reduce the fraction
    g := gcd(abs(num), abs(den))
    q.num = num / g
    q.den = den / g
}

func gcd(a, b int) int {
    for b != 0 {
        a, b = b, a%b
    }
    return a
}

func abs(x int) int {
    if x < 0 {
        return -x
    }
    return x
}

// --- Additive interface ---
func (a Q) _add(b Q) Q {
    return *make(Q, a.num*b.den + b.num*a.den, a.den*b.den)
}

func (a Q) _neg() Q {
    return *make(Q, -a.num, a.den)
}

func (a Q) Zero() Q {
    return *make(Q, 0, 1)
}

// --- Multiplicative interface ---
func (a Q) _mul(b Q) Q {
    return *make(Q, a.num*b.num, a.den*b.den)
}

func (a Q) One() Q {
    return *make(Q, 1, 1)
}

// --- MulInvertible interface (field-specific) ---
func (a Q) Reciprocal() Q {
    if a.num == 0 {
        panic("cannot invert zero")
    }
    return *make(Q, a.den, a.num)
}

func (a Q) IsZero() bool {
    return a.num == 0
}

// --- Division implemented via multiplicative inverse ---
func (a Q) _div(b Q) Q {
    return a * b.Reciprocal()  // a × b⁻¹
}

// --- Subtraction ---
func (a Q) _sub(b Q) Q {
    return a + (-b)
}

// --- Comparison ---
func (a Q) _eq(b Q) bool {
    return a.num == b.num && a.den == b.den
}

func (a Q) _lt(b Q) bool {
    return a.num*b.den < b.num*a.den
}

// --- String representation ---
func (a Q) String() string {
    if a.den == 1 {
        return fmt.Sprintf("%d", a.num)
    }
    return fmt.Sprintf("%d/%d", a.num, a.den)
}

// ============================================
// Generic field operations
// ============================================

// Division applicable to any Field
func Divide[T Field[T]](a, b T) T {
    if b.IsZero() {
        panic("division by zero")
    }
    return a * b.Reciprocal()
}

// Solve a linear equation ax + b = 0, returning x = -b/a
func SolveLinear[T Field[T]](a, b T) T {
    return Divide(-b, a)
}

func main() {
    a := Z(3)
    b := Z(5)

    // Operator overloading makes the syntax natural
    fmt.Println("3 + 5 =", a + b)        // 8
    fmt.Println("3 × 5 =", a * b)        // 15
    fmt.Println("-3 =", -a)              // -3
    fmt.Println("3 - 5 =", a - b)        // -2

    // Generic power function
    fmt.Println("3^4 =", Pow(a, 4))      // 81

    // Generic summation
    fmt.Println("Σ(1,2,3,4,5) =", Sum(Z(1), Z(2), Z(3), Z(4), Z(5)))  // 15

    // Construct rational numbers using make
    half := *make(Q, 1, 2)
    third := *make(Q, 1, 3)

    fmt.Println("1/2 + 1/3 =", half + third)       // 5/6
    fmt.Println("1/2 × 1/3 =", half * third)       // 1/6
    fmt.Println("1/2 ÷ 1/3 =", half / third)       // 3/2
    fmt.Println("(1/2)⁻¹ =", half.Reciprocal())    // 2

    // Reduction check
    six_nine := *make(Q, 6, 9)
    fmt.Println("6/9 =", six_nine)                 // 2/3

    // Solve the equation 3x + 6 = 0
    qa := *make(Q, 3)
    qb := *make(Q, 6)
    x := SolveLinear(qa, qb)
    fmt.Println("3x + 6 = 0 → x =", x)             // -2
}
```

## 8. Static Dispatch

MyGO generics introduce a **static dispatch generics** mechanism:  
by adding the `static` keyword before a type parameter, you can **force monomorphization** at compile time, similar to how generics are instantiated in Rust or C++.

### 8.1 Syntax

- **Generic functions**:

```go
func MyFunc[static T, U any](t T, u U) {}
````

* **Generic structs / types**:

```go
type Box[static T any, U any] struct {
    t T
    u U
}
```

> `static` is a **contextual keyword**: it is only valid inside the `[...]` type parameter list.

### 8.2 Group Propagation Rules

Go allows *constraint grouping* in type parameter lists, for example `[T, U any]`, where `T` and `U` share the same constraint `any`.

MyGO’s `static` follows a **group-based propagation rule**:

* `func F[static T, U any]()`
  Since `T` and `U` belong to the same constraint group (`any`), **both `T` and `U` are treated as static**.

* `func F[static T Interface1, U Interface1]()`
  Here `T` and `U` are in different constraint groups, so **only `T` is static**.

When printing / formatting type parameter lists, `static` is shown only once at the **head of each constraint group** (e.g. `static T, U any`), but **semantically all parameters in the group are static**.

### 8.3 Semantics: “No Dict” for Static, Dict Still Allowed for Non-Static

MyGO currently supports **two generic dispatch mechanisms simultaneously**:

* **Static type parameters**:

  * Specialized implementations are generated at instantiation time (monomorphization)
  * **No runtime dictionary (`dict`) is generated for static parameters**
  * In the backend IR, static parameters use **concrete native types**
    (they are not shapified into `go.shape.*`)

* **Non-static type parameters**:

  * Continue to use the existing *shape + runtime dictionary* mechanism
    (to reduce code size bloat)

Therefore, **mixed mode is allowed**.
In `[static T, U any]`, `T` is monomorphized, while `U` may still require a dictionary
(e.g. for interface method dispatch, reflection, or certain conversions).

### 8.4 Static Arguments Must Be Concrete Types

A `static` type parameter **must be instantiated with a concrete type**:

* It cannot be another type parameter
* It cannot contain type parameters

Otherwise, true monomorphization is impossible.

Example (this will fail):

```go
func F[static T any]() {}

func G[U any]() {
    F[U]() // ❌ U is a type parameter, cannot be used to instantiate static T
}
```

### 8.5 Cross-Package Behavior: Specialization in the Caller Package

For cross-package calls, MyGO generates the specialized implementation in the **caller’s package**, using a stable hash-mangled symbol name to avoid excessively long symbols:

```text
mygo_ + PkgPath + "." + FuncName + "_STA_" + SignatureHash
```

Where `SignatureHash` is a hash of the **canonicalized full names of the type arguments**
(currently: first 8 hex digits of SHA256), providing both stability and controlled symbol length.

> Note: `runtime.FuncForPC` may elide generic instantiations (displaying `[...]`),
> so it is not reliable for verifying final symbol names.

### 8.6 Usage Example

This feature is primarily intended for **performance-critical generic code**.
Below is a benchmark demonstrating the performance difference.

```go
package main

import (
    "fmt"
    "time"
)

// Define a simple interface
type Number interface {
    Get() int
}

// Define a concrete struct
type MyInt struct {
    Val int
}

// Implement the interface method (very small, ideal for inlining)
//
// Note: a pointer receiver is required to trigger Go’s standard
// "shape sharing + dictionary lookup" mechanism.
func (m *MyInt) Get() int {
    return m.Val
}

// ---------------------------------------------------------
// 1. Standard Go Generics
// ---------------------------------------------------------
//
// For pointer types T, Go 1.18+ uses GCShape (go.shape.*uint8)
// and performs method lookup via a runtime dictionary.
// This prevents inlining and introduces indirect call overhead.
func SumStandard[T Number](data []T) int {
    sum := 0
    for _, v := range data {
        sum += v.Get() // Dictionary lookup + indirect call
    }
    return sum
}

// ---------------------------------------------------------
// 2. MyGO Static Dispatch
// ---------------------------------------------------------
//
// Using the `static` keyword forces monomorphization.
// The compiler generates a specialized version for *MyInt.
// v.Get() becomes a direct call and may even be inlined
// into a single ADD instruction.
func SumStatic[static T Number](data []T) int {
    sum := 0
    for _, v := range data {
        sum += v.Get() // Direct call (high chance of inlining)
    }
    return sum
}

// ---------------------------------------------------------
// 3. Non-generic Baseline
// ---------------------------------------------------------
//
// Hand-written concrete implementation, representing
// the theoretical performance upper bound.
func SumBaseline(data []*MyInt) int {
    sum := 0
    for _, v := range data {
        sum += v.Get()
    }
    return sum
}

func main() {
    const N = 100_000_000 // 100 million iterations to amplify tiny overheads

    fmt.Printf("Preparing data: %d elements...\n", N)

    // Initialize slice
    data := make([]*MyInt, N)
    for i := 0; i < N; i++ {
        data[i] = &MyInt{Val: 1}
    }

    // Warm-up (avoid CPU frequency scaling effects)
    SumBaseline(data[:1000])

    // --- Test 1: Baseline ---
    start := time.Now()
    resBase := SumBaseline(data)
    durBase := time.Since(start)
    fmt.Printf("[Baseline]   Non-generic native: %v (Result: %d)\n", durBase, resBase)

    // --- Test 2: Standard Go Generics ---
    start = time.Now()
    resStd := SumStandard(data) // T inferred as *MyInt
    durStd := time.Since(start)
    fmt.Printf("[Standard]   Go generics:        %v (Result: %d)\n", durStd, resStd)

    // --- Test 3: MyGO Static ---
    start = time.Now()
    resSta := SumStatic(data) // static T inferred as *MyInt
    durSta := time.Since(start)
    fmt.Printf("[MyGO Static] Static dispatch:   %v (Result: %d)\n", durSta, resSta)

    // --- Analysis ---
    fmt.Println("--------------------------------------------------")
    fmt.Printf("Speedup (Static vs Standard): %.2fx faster\n",
        float64(durStd)/float64(durSta))
    fmt.Printf("Overhead (Static vs Baseline): %.2f%% (closer to 0 is better)\n",
        (float64(durSta)-float64(durBase))/float64(durBase)*100)
}
```

---

## Notes

### Constructor Decorators

You can decorate `_init` constructors, but there are specific signature requirements.

**Requirements:**
- The decorator's function signature must match the original signature of the _init method (before any internal rewriting).
- The decorator must accept and return a function type where the return type includes *TypeName.

**Correct Example:**

```go
// Decorator signature: accepts and returns func(string, int) *Server
func logger(f func(string, int) *Server) func(string, int) *Server {
    return func(host string, port int) *Server {
        fmt.Println("Creating server:", host, port)
        return f(host, port)
    }
}

type Server struct {
    host string
    port int
}

@logger
func (s *Server) _init(host string, port int) {
    s.host = host
    s.port = port
}

func main() {
    s := make(Server, "localhost", 8080)
    // Output: Creating server: localhost 8080
    fmt.Printf("%s:%d\n", s.host, s.port)
}
```

### Method Overloading vs Default Args

Ambiguity can arise when a method has both overloads and default arguments. MyGO resolves this using **Declaration Order Priority**.


**Ambiguity Example:**


```go
type DataStore struct {
    intData    map[string]int
    stringData map[string]string
}

// Store Int
func (ds *DataStore) Set(key string, value int) {
    if ds.intData == nil {
        ds.intData = make(map[string]int)
    }
    ds.intData[key] = value
}

// Store String
func (ds *DataStore) Set(key string, value string) {
    if ds.stringData == nil {
        ds.stringData = make(map[string]string)
    }
    ds.stringData[key] = value
}

// First Get method: handles int with a default argument
func (ds *DataStore) Get(key string, defaultValue int = 0) int {
    if v, ok := ds.intData[key]; ok {
        return v
    }
    return defaultValue
}

// Second Get method: handles string with a default argument
func (ds *DataStore) Get(key string, defaultValue string = "Unknown") string {
    if v, ok := ds.stringData[key]; ok {
        return v
    }
    return defaultValue
}

func main() {
    store := &DataStore{}
    store.Set("age", 25)
    
    // ⚠️ Ambiguous Case: Only one argument provided
    // This calls the first declared method (the int version)
    result := store.Get("someKey") 

    fmt.Println(result) // Output: 0

    // ✅ Best Practice: Pass full arguments to avoid ambiguity
    intResult := store.Get("age", 0)           // Calls int version
    strResult := store.Get("name", "Unknown")  // Calls string version
    fmt.Println(intResult, strResult)
}
```

**Advice:**
- Avoid using default arguments within overloaded methods whenever possible.
- If you must use them, explicitly pass all arguments at the call site to ensure the correct overload is selected.
- Alternatively, use distinct method names (e.g., `GetInt`, `GetString`).

## Build and Use

1. Clone and Build:
```bash
cd src
GOROOT_BOOTSTRAP=/usr/local/go ./make.bash
```

2. Run with MyGO:
```bash
GOROOT=/path/to/mygo /path/to/mygo/bin/go run your_file.go
```