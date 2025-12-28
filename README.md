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

**⚠️ Note: `select` statement not supported yet!**

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

Native constructors are also supported:

```go
// slice implies _init(pos int) and _init(pos int, cap int)
// map/chan implies _init() and _init(pos int)
```

**⚠️ Warning ⚠️**
`type Name float64` declarations can be overloaded, but it is **not recommended**. It causes boxing of the primitive type, hurting performance, and the type is no longer treated as its underlying type.

## 7. Algebraic Data Types (Experimental)

### Enums

MyGO supports `enum` (Tagged Union / ADT). An enum consists of multiple **Variants**.

#### 7.1 Defining Enums

```go
// Unit variants
type Color enum {
    Red
    Green
    Blue
}

// Payload variants (Tuple)
type Shape enum {
    Circle(float64)
    Rect(float64, float64)
    Point
}

// Recursive Enum
type List enum {
    Cons(int, List)
    Nil
}

// Generic Enum
type Option[T any] enum {
    Some(T)
    None
}
```

#### 7.2 Constructing Values

```go
c1 := Color.Red             // unit: no parens
s1 := Shape.Circle(1.5)     // payload: args required
s2 := Shape.Rect(3, 4)
o1 := Option[int].Some(42)
```

#### 7.3 Pattern Matching

Use `switch` to match structure at compile time.

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

**Wildcard**: Use `_` to ignore fields.

##### Exhaustiveness Checking

MyGO checks if all variants are handled in a `switch`. If not (and no default), the compiler errors:
`error: enum match on Shape is not exhaustive (missing: Point)`

#### 7.4 Generics + Shape (GC Shape) & Storage

Enums are lowered to Go `struct` + `unsafe` constructors.
*   **Pure Values**: Stored in `_stack [N]byte` (Stack inline).
*   **Pointer Payloads**: Stored in `_heap unsafe.Pointer`.
*   **Generics**: Strategy depends on instantiated type.

#### 7.5 if/for Pattern Matching

##### if match

```go
opt := Option[int].Some(42)

if Option.Some(x) := opt {
    fmt.Println("value:", x)
} else {
    fmt.Println("no value")
}

// Guard
if Option.Some(x) := opt; x > 0 {
    fmt.Println("positive:", x)
}
```

##### for match

Loops while pattern matches:

```go
shape := Shape.Circle(1.5)
for Shape.Circle(r) := shape {
    fmt.Println("radius:", r)
    shape = Shape.Point // Exit loop
}
```

#### 7.6 Enum Magic Functions

Enums can act as **Dynamic Dispatchers** by implementing magic methods (`_add`, etc.).

```go
type Value enum {
    Integer(int)
    Float(float64)
}

// Value + Value
func (lhs Value) _add(rhs Value) Value {
    switch lhs {
    case Value.Integer(a):
        switch rhs {
        case Value.Integer(b):
            return Value.Integer(a + b)
        case Value.Float(b):
            return Value.Float(float64(a) + b)
        }
    // ... handle other cases
    }
    panic("unsupported")
}
```

#### 7.7 Enum Usage Examples

##### Option[T]

```go
// Option implementation with _eq and _div overloading...
// Allows code like:
// x := Option[int].Some(10) / Option[int].Some(2)
```

##### Result[T, E]

Structured error handling.

```go
type Result[T any, E error] enum {
    Ok(T)
    Err(E)
}

func safeDiv(a, b int) Result[int, error] {
    if b == 0 {
        return Result[int, error].Err(errors.New("division by zero"))
    }
    return Result[int, error].Ok(a / b)
}
```

##### Monad Style

Supports `Map`, `AndThen`, `UnwrapOr` patterns easily.

---

## Notes

### Constructor Decorators

You can decorate `_init`, but:
*   Decorator signature must match `_init` args.
*   Decorator must return a function that returns `*TypeName`.

### Method Overloading vs Default Args

Ambiguity can arise. Compiler prioritizes **Declaration Order**.
**Advice**: Avoid mixing overloading and default args, or use explicit arguments.

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