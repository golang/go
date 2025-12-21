# It's MyGO!!!!!

本仓库为GO(fork)的非官方个人改造版本，因此叫`MyGO`.

![MyGO image](img/mygo1.png)
*MyGO image by Gemini/Nano-banana-pro*

![MyGO Logo](img/mygo2.jpg)
*MyGO的Logo*

--------------

本仓库已经实现的特性有以下七个

## 1. 装饰器

装饰器允许你在函数声明时使用 `@decorator` 语法来包装函数。

### 1.1 基础装饰器（无参数函数）

```go
func logger(f func()) func() {
    return func() {
        fmt.Println("开始执行")
        f()
        fmt.Println("执行完毕")
    }
}

@logger
func sayHello() {
    fmt.Println("Hello, MyGO!")
}

func main() {
    sayHello()
    // 输出:
    // 开始执行
    // Hello, MyGO!
    // 执行完毕
}
```

### 1.2 装饰带参数的函数

装饰器可以包装任意签名的函数，包括带参数和返回值的函数。

```go
// 计时装饰器 - 装饰带参数和返回值的函数
func timeit(f func(int, int) int) func(int, int) int {
    return func(a, b int) int {
        start := time.Now()
        result := f(a, b)
        elapsed := time.Since(start)
        fmt.Printf("执行耗时: %v\n", elapsed)
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
    fmt.Println("结果:", result)
    // 输出:
    // 执行耗时: 100.xxxms
    // 结果: 8
}
```

### 1.3 带参数的装饰器

装饰器本身也可以接受参数，实现更灵活的装饰逻辑。

```go
// 带参数的装饰器：重复执行 n 次
func repeat(f func(), n int) func() {
    return func() {
        for i := 0; i < n; i++ {
            fmt.Printf("第 %d 次执行:\n", i+1)
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
    // 输出:
    // 第 1 次执行:
    // Hello, MyGO!
    // 第 2 次执行:
    // Hello, MyGO!
    // 第 3 次执行:
    // Hello, MyGO!
}
```

### 1.4 多个装饰器链式调用

可以在同一个函数上应用多个装饰器，按从上到下的顺序执行。

```go
func logger(f func()) func() {
    return func() {
        fmt.Println("[LOG] 开始执行")
        f()
        fmt.Println("[LOG] 执行完毕")
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
    // 输出:
    // [LOG] 开始执行
    // === START ===
    // Hello, MyGO!
    // === END ===
    // [LOG] 执行完毕
}
```

### 1.5 错误处理装饰器

```go
// 错误恢复装饰器
func recover_errors(f func()) func() {
    return func() {
        defer func() {
            if r := recover(); r != nil {
                fmt.Printf("捕获到错误: %v\n", r)
            }
        }()
        f()
    }
}

@recover_errors
func riskyOperation() {
    fmt.Println("执行危险操作...")
    panic("出错了！")
}

func main() {
    riskyOperation()
    fmt.Println("程序继续运行")
    // 输出:
    // 执行危险操作...
    // 捕获到错误: 出错了！
    // 程序继续运行
}
```

### 1.6 权限验证装饰器

```go
var currentUser = "admin"

func requireAuth(f func(string), role string) func(string) {
    return func(user string) {
        if user != role {
            fmt.Printf("权限不足：需要 %s 角色\n", role)
            return
        }
        f(user)
    }
}

@requireAuth("admin")
func deleteUser(user string) {
    fmt.Printf("用户 %s 已删除\n", user)
}

func main() {
    deleteUser("admin")  // 输出: 用户 admin 已删除
    deleteUser("guest")  // 输出: 权限不足：需要 admin 角色
}
```

### 1.7 适配不同参数数量的装饰器的最佳实践

可以使用`reflect`来实现

```go
package main

import (
	"fmt"
	"reflect"
	"time"
)

// 通用计时装饰器
// T 可以是任意函数类型
func TimeIt[T any](f T) T {
	// 1. 获取函数的反射值和类型
	fnVal := reflect.ValueOf(f)
	fnType := fnVal.Type()

	// 确保传入的是一个函数
	if fnType.Kind() != reflect.Func {
		panic("TimeIt decorator requires a function")
	}

	// 2. 使用 MakeFunc 创建一个新的函数
	// MakeFunc 创建一个具有给定类型 fnType 的新函数
	// 当该函数被调用时，它会执行传入的匿名函数 (args []reflect.Value) []reflect.Value
	wrapper := reflect.MakeFunc(fnType, func(args []reflect.Value) []reflect.Value {
		start := time.Now()

		// 3. 调用原始函数
		// 注意：这里会有一定的反射性能开销，但在大多数业务逻辑中可以忽略
		results := fnVal.Call(args)

		elapsed := time.Since(start)
		fmt.Printf("==> 执行耗时: %v\n", elapsed)

		return results
	})

	// 4. 将创建的反射值转换回 T 类型并返回
	return wrapper.Interface().(T)
}

// 装饰器：两个参数
@TimeIt
func add(x, y int) int {
	time.Sleep(50 * time.Millisecond)
	return x + y
}

// 装饰器：一个参数
@TimeIt
func inverse(x int) int {
	time.Sleep(50 * time.Millisecond)
	return 100 / x
}

// 装饰器：无参数，无返回值
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


## 2. 函数默认参数

支持为函数参数设置默认值，调用时可省略有默认值的参数。

**规则**：默认值只能从后往前设置（即有默认值的参数必须在参数列表末尾）

```go
// 所有参数都有默认值
func greet(name string = "World", greeting string = "Hello") {
    fmt.Printf("%s, %s!\n", greeting, name)
}

// 部分参数有默认值
func calculate(x int, y int = 10, z int = 5) int {
    return x + y + z
}

func main() {
    greet()                    // 输出: Hello, World!
    greet("MyGO")              // 输出: Hello, MyGO!
    greet("MyGO", "Hi")        // 输出: Hi, MyGO!
    
    fmt.Println(calculate(1))        // 输出: 16 (1 + 10 + 5)
    fmt.Println(calculate(1, 2))     // 输出: 8  (1 + 2 + 5)
    fmt.Println(calculate(1, 2, 3))  // 输出: 6  (1 + 2 + 3)
}
```

## 3. 可选链

使用 `?.` 操作符进行空安全的字段访问和方法调用，避免 nil 指针错误。

*重要! 可选链返回的都是指针类型，需要转回值类型！————因为要支持nil*

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
    
    // 传统方式需要检查 nil
    if user2.Profile != nil {
        fmt.Println(user2.Profile.Email)
    }
    
    // 使用可选链，自动处理 nil
    email1 := user1?.Profile?.Email  // "alice@example.com"
    email2 := user2?.Profile?.Email  // nil (不会 panic)
    
    fmt.Println("email1:", *email1)  // alice@example.com
    fmt.Println("email2:", email2)  // <nil>
}
```

## 4. 三元表达式

支持简洁的条件表达式 `condition ? trueValue : falseValue`。

**注意** 三元表达式的True分支和False分支，必须保证同类型，否则报错！

```go
func main() {
    age := 18
    
    // 传统 if-else
    var status string
    if age >= 18 {
        status = "成年"
    } else {
        status = "未成年"
    }
    
    status = age >= 18 ? "成年" : "未成年"
    fmt.Println(status)  // 输出: 成年
    
    fmt.Println(age >= 18 ? "成年" : "未成年")  // 输出: 成年
    
    // 嵌套三元表达式
    score := 85
    grade := score >= 90 ? "A" : score >= 80 ? "B" : score >= 60 ? "C" : "D"
    fmt.Println(grade)   // 输出: B
    
    // 用于数值类型
    max := 10 > 5 ? 10 : 5
    fmt.Println(max)  // 输出: 10
}
```

### 与可选链联动的优雅用法

MyGO 支持Elvis运算符`x?:y`，可以通过Elvis运算符进行空值运算，他是三目运算符的一种特例

由于可选链返回的都是指针类型，需要转回值类型。但是对nil直接取值会导致panic，于是建议使用下面的方案写


```go
func main() {
    user1 := &User{Name: "Alice", Profile: &Profile{Email: "alice@example.com", Age: 25}}
    user2 := &User{Name: "Bob", Profile: nil}
    
    // 传统方式需要检查 nil
    if user2.Profile != nil {
        fmt.Println(user2.Profile.Email)
    }
    
    // 使用三元表达式 + 可选链，自动处理 nil
    email1 := (user1?.Profile?.Email!=nil) ? *user1?.Profile?.Email : ""  
    email2 := (user2?.Profile?.Email!=nil) ? *user2?.Profile?.Email : ""  
    
    fmt.Println("email1:", email1)  // alice@example.com
    fmt.Println("email2:", email2)  // ""
    
    // 上述复杂的三元表达式可写作 新语法：x?:y  等价于  (x!=nil) ? *x : y
    email1 := user1?.Profile?.Email?:""
    email2 := user2?.Profile?.Email?:""
    
    fmt.Println("email1:", email1)  // alice@example.com
    fmt.Println("email2:", email2)  // ""

}
```

## 5. 结构体方法重载

支持为结构体定义多个同名方法，只要参数类型不同即可。编译器会根据调用时的参数类型自动选择正确的方法。

### 5.1 基础用法

```go
type Calculator struct{}

// 整数加法
func (c *Calculator) Add(a int, b int) int {
    return a + b
}

// 浮点数加法
func (c *Calculator) Add(a float64, b float64) float64 {
    return a + b
}

// 字符串拼接
func (c *Calculator) Add(a string, b string) string {
    return a + b
}

func main() {
    calc := &Calculator{}
    
    intResult := calc.Add(1, 2)           // 调用 int 版本，返回 int
    floatResult := calc.Add(1.5, 2.5)     // 调用 float64 版本，返回 float64
    strResult := calc.Add("Hello", "GO")  // 调用 string 版本，返回 string
    
    fmt.Printf("int: %T = %v\n", intResult, intResult)       // int: int = 3
    fmt.Printf("float64: %T = %v\n", floatResult, floatResult) // float64: float64 = 4.0
    fmt.Printf("string: %T = %v\n", strResult, strResult)    // string: string = HelloGO
}
```

### 5.2 不同参数数量

```go
type Greeter struct{}

// 无参数版本
func (g *Greeter) SayHello() string {
    return "Hello, World!"
}

// 单参数版本
func (g *Greeter) SayHello(name string) string {
    return "Hello, " + name + "!"
}

// 双参数版本
func (g *Greeter) SayHello(firstName string, lastName string) string {
    return "Hello, " + firstName + " " + lastName + "!"
}

func main() {
    g := &Greeter{}
    
    fmt.Println(g.SayHello())                    // 输出: Hello, World!
    fmt.Println(g.SayHello("Alice"))             // 输出: Hello, Alice!
    fmt.Println(g.SayHello("Bob", "Smith"))      // 输出: Hello, Bob Smith!
}
```

### 5.3 不同返回值类型

```go
type Processor struct{}

// 处理整数，返回平方值
func (p *Processor) Process(x int) int {
    return x * x
}

// 处理浮点数，返回平方根
func (p *Processor) Process(x float64) float64 {
    return math.Sqrt(x)
}

// 处理字符串，返回长度
func (p *Processor) Process(s string) int {
    return len(s)
}

func main() {
    proc := &Processor{}
    
    intResult := proc.Process(5)        // int: 25
    floatResult := proc.Process(16.0)   // float64: 4.0
    lenResult := proc.Process("MyGO")   // int: 4
    
    // 无需类型断言！编译器知道每个返回值的确切类型
    fmt.Printf("%T: %v\n", intResult, intResult)
    fmt.Printf("%T: %v\n", floatResult, floatResult)
    fmt.Printf("%T: %v\n", lenResult, lenResult)
}
```

### 5.4 实际应用示例

```go
type DataStore struct {
    intData    map[string]int
    stringData map[string]string
}

// 存储整数
func (ds *DataStore) Set(key string, value int) {
    if ds.intData == nil {
        ds.intData = make(map[string]int)
    }
    ds.intData[key] = value
}

// 存储字符串
func (ds *DataStore) Set(key string, value string) {
    if ds.stringData == nil {
        ds.stringData = make(map[string]string)
    }
    ds.stringData[key] = value
}

// 获取整数
func (ds *DataStore) Get(key string, defaultValue int) int {
    if v, ok := ds.intData[key]; ok {
        return v
    }
    return defaultValue
}

// 获取字符串
func (ds *DataStore) Get(key string, defaultValue string) string {
    if v, ok := ds.stringData[key]; ok {
        return v
    }
    return defaultValue
}

func main() {
    store := &DataStore{}
    
    // 存储不同类型的数据
    store.Set("age", 25)
    store.Set("name", "Alice")
    
    // 获取数据，类型安全
    age := store.Get("age", 0)           // int
    name := store.Get("name", "Unknown") // string
    
    fmt.Printf("Age: %d, Name: %s\n", age, name) // Age: 25, Name: Alice
}
```

## 6. 魔法函数(实验特性)

注意，标记为实验特性的可能有BUG，慎用 (魔法方法所有的特性主要为未来的向量计算作准备)

### 构造函数

使用 `make(TypeName, args...)` 语法创建结构体实例，支持自定义初始化逻辑。

**规则**：
- 构造方法必须命名为 `_init`
- 必须是指针接收器方法（`func (t *Type) _init(...)`）
- 不需要手动写返回值（编译器自动添加）
- 支持重载（不同参数类型）和默认参数

#### 基础用法

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
    fmt.Println(p.name, p.age) // 输出: Alice 25
}
```

#### 构造函数重载

```go
type Database struct {
    host     string
    port     int
    username string
}

// 通过端口号初始化
func (d *Database) _init(host string, port int) {
    d.host = host
    d.port = port
    d.username = "admin"
}

// 通过用户名初始化
func (d *Database) _init(host string, username string) {
    d.host = host
    d.port = 3306 // 默认端口
    d.username = username
}

func main() {
    db1 := make(Database, "localhost", 3306)   // 调用第一个 _init
    db2 := make(Database, "localhost", "root") // 调用第二个 _init
    
    fmt.Println(db1.host, db1.port, db1.username) // localhost 3306 admin
    fmt.Println(db2.host, db2.port, db2.username) // localhost 3306 root
}
```

#### 构造函数 + 默认参数

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
    s1 := make(Server, "localhost")         // 使用默认端口 8080
    s2 := make(Server, "0.0.0.0", 3000)     // 指定端口
    
    fmt.Printf("%s:%d\n", s1.host, s1.port) // localhost:8080
    fmt.Printf("%s:%d\n", s2.host, s2.port) // 0.0.0.0:3000
}
```

### 索引运算符重载 (_getitem / _setitem)

MyGo 支持通过 _getitem 和 _setitem 方法实现自定义类型的索引操作

#### 规则描述

1. 有逗号 → 强制匹配 []T 参数
2. 无逗号 → 优先匹配 T 参数，fallback 到 []T

#### 基础示例

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

// _getitem: 支持 matrix[row, col] 语法
func (m *Matrix) _getitem(indices1 []int, indices2 []int) int {
	row, col := indices1[0], indices2[0]
	return m.data[row][col]
}

// _setitem: 支持 matrix[row, col] = value 语法
func (m *Matrix) _setitem(indices1 []int, indices2 []int, value int) {
	row, col := indices1[0], indices2[0]
	m.data[row][col] = value
}

func main() {
	// 使用 make 创建 Matrix，自动调用 _init
	m := make(Matrix, 3, 3)
	
	// 设置值 - 调用 _setitem
	m[0, 0] = 1
	m[1, 1] = 5
	m[2, 2] = 9
	
	// 获取值 - 调用 _getitem
	fmt.Println(m[0, 0])  // 输出: 1
	fmt.Println(m[1, 1])  // 输出: 5
	fmt.Println(m[2, 2])  // 输出: 9
}
```

#### 逗号语法 与 冒号语法

MyGO 可以区分 逗号分隔 和 冒号切片 两种索引语法：

```go
package main

import "fmt"

type NDArray struct {
	data []int
}

func (a *NDArray) _init(data []int) {
	a.data = data
}

// 冒号语法: arr[start:end] → 传入普通参数
func (a *NDArray) _getitem(args ...int) []int {
	fmt.Printf("切片访问: %v\n", args)
	start, end := args[0], args[1]
	return a.data[start:end]
}

// 逗号语法: arr[i, j, k] → 传入切片参数
func (a *NDArray) _getitem(indices ...[]int) int {
	fmt.Printf("多维索引: %v\n", indices)
	// 处理多维索引逻辑...
	return 0
}

func main() {
	arr := make(NDArray, []int{1, 2, 3, 4, 5})
	
	// 冒号语法 - 匹配 ...int 版本
	_ = arr[1:3]      // 输出: 切片访问: [1, 3]
	
	// 逗号语法 - 匹配 ...[]int 版本  
	_ = arr[1, 2]     // 输出: 多维索引: [[1], [2]]
	_ = arr[1:2, 3:4] // 输出: 多维索引: [[1, 2], [3, 4]]
}
```

#### 用法案例

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

// _getitem: 支持 person["name"] 语法
func (p *Person) _getitem(name string) string {
	if value, ok := p.data[name]; ok {
		return value
	}
	return "not found"
}

// _setitem: 支持 person["name"] = value 语法 ⚠️ 注意 value 在前面 key 在后面
func (p *Person) _setitem(value string, name string) {
	p.data[name] = value
}

func main() {
	// 使用 make 创建 Person，自动调用 _init
	person := make(Person)

	// 获取值 - 调用 _getitem
	fmt.Println("Name:", person["name"]) // 输出: Name: Alice
	fmt.Println("Age:", person["age"])   // 输出: Age: 25
	fmt.Println("City:", person["city"]) // 输出: City: Beijing

	// 设置值 - 调用 _setitem
	person["name"] = "Bob"
	person["country"] = "China"

	// 再次获取值
	fmt.Println("Updated Name:", person["name"]) // 输出: Updated Name: Bob
	fmt.Println("Country:", person["country"])   // 输出: Country: China
	fmt.Println("Unknown:", person["unknown"])   // 输出: Unknown: not found
}
```

### 算数运算符重载

MyGO 支持通过 `_add`、`_sub` 等一系列特定命名的方法，实现对标准算术运算符（`+`, `-`, `*`, `/` 等）的重载。

**规则描述**

编译器在处理二元运算（如 `a + b`）时，遵循以下解析顺序：

1. 正向优先 (*Forward*)：优先尝试调用左操作数的方法（如 `a._add(b)`）。

2. 反向回退 (*Fallback*)：如果正向方法不存在或不匹配，尝试调用右操作数反向方法（如 `b._radd(a)`）。

3. 单目特例：自增 (`++`) 和自减 (`--`) 仅支持正向调用，不支持反向。

##### 二元运算符重载一览表

| 运算符 | 正向方法 (主) | 反向方法 (备选) | 备注 |
| :---: | :--- | :--- | :--- |
| `+` | `func _add(b T) T` | `func _radd(a T) T` | 加法 |
| `-` | `func _sub(b T) T` | `func _rsub(a T) T` | 减法 |
| `*` | `func _mul(b T) T` | `func _rmul(a T) T` | 乘法 |
| `/` | `func _div(b T) T` | `func _rdiv(a T) T` | 除法 |
| `%` | `func _mod(b T) T` | `func _rmod(a T) T` | 取模 |
| `++` | `func _inc()` | N/A | 自增 (无返回值) |
| `--` | `func _dec()` | N/A | 自减 (无返回值) |

##### 一元运算符重载一览表

|  运算符 | 正向方法               | 反向方法 | 备注              |
| :--: | :----------------- | :--- | :-------------- |
| `+a` | `func _pos() T`    | N/A  | 正号（通常返回自身或拷贝） |
| `-a` | `func _neg() T`    | N/A  | 取负            |
| `^a` | `func _invert() T` | N/A  | 按位取反          |

##### 比较运算符重载一览表

|  运算符 | 正向方法                 | 镜像回退方法 | 备注   |
| :--: | :------------------- | :----- | :--- |
| `==` | `func _eq(v T) bool` | `_eq`  | 相等   |
| `!=` | `func _ne(v T) bool` | `_ne`  | 不等   |
|  `>` | `func _gt(v T) bool` | `_lt`  | 大于   |
| `>=` | `func _ge(v T) bool` | `_le`  | 大于等于 |
|  `<` | `func _lt(v T) bool` | `_gt`  | 小于   |
| `<=` | `func _le(v T) bool` | `_ge`  | 小于等于 |

**镜像回退规则（非常重要）**

当表达式 a OP b 中：

a 未实现 对应的比较方法

编译器会自动尝试调用 右操作数的镜像方法

例如

| 表达式      | 回退尝试       |
| -------- | ---------- |
| `a < b`  | `b._gt(a)` |
| `a <= b` | `b._ge(a)` |
| `a > b`  | `b._lt(a)` |
| `a >= b` | `b._le(a)` |
| `a == b` | `b._eq(a)` |
| `a != b` | `b._ne(a)` |

**⚠️ 指针类型特别警告**

如果指针类型重载了 `==` 或 `!=`，将覆盖 Go 原生的指针地址比较语义

编译器将警告这种行为(但编译依然进行)

` warning: *T defines _eq (== overload) or _ne (!= overload), which overrides the native pointer == or != semantics for this type`

##### 位运算符重载一览表

|  运算符 | 正向方法                    | 反向方法                     | 备注   |
| :--: | :---------------------- | :----------------------- | :--- |
| `\|` | `func _or(v T) T`       | `func _ror(v T) T`       | 按位或  |
|  `&` | `func _and(v T) T`      | `func _rand(v T) T`      | 按位与  |
|  `^` | `func _xor(v T) T`      | `func _rxor(v T) T`      | 按位异或 |
| `<<` | `func _lshift(v T) T`   | `func _rlshift(v T) T`   | 左移   |
| `>>` | `func _rshift(v T) T`   | `func _rrshift(v T) T`   | 右移   |
| `&^` | `func _bitclear(v T) T` | `func _rbitclear(v T) T` | 按位清零 |

##### 数据流运算符重载一览表

|    运算符   |  位置 | 方法                | 反向方法 | 备注 |
| :------: | :-: | :---------------- | :--- | :- |
|   `<-a`  |  前缀 | `func _recv() T`  | N/A  | 接收 |
| `a <- v` |  中缀 | `func _send(v T)` | N/A  | 发送 |

**⚠️ 注意：暂不支持select语句！**

##### 复合赋值自动展开规则

如果某类型实现了某个操作符，则：

| 语法        | 自动展开         |
| --------- | ------------ |
| `a += b`  | `a = a + b`  |
| `a -= b`  | `a = a - b`  |
| `a &= b`  | `a = a & b`  |
| `a &^= b` | `a = a &^ b` |
| `a <<= b` | `a = a << b` |

#### 向量运算

这个特性最常见的场景就是`向量计算`

```go
package main

import "fmt"

// Vector 支持正向 + 和 ++
type Vector struct {
    x, y int
}

// _add: 对应 a + b
func (v *Vector) _add(other *Vector) *Vector {
    return &Vector{x: v.x + other.x, y: v.y + other.y}
}

// _sub: 对应 a - b
func (v *Vector) _sub(other *Vector) *Vector {
    return &Vector{x: v.x - other.x, y: v.y - other.y}
}

// _inc: 对应 v++
func (v *Vector) _inc() {
    v.x++
    v.y++
}

func main() {
    v1 := &Vector{x: 1, y: 1}
    v2 := &Vector{x: 2, y: 3}

    // 基础加减
    v3 := v1 + v2       // 调用 v1._add(v2)
    fmt.Println(v3)     // 输出: &{3 4}
    
    // 链式运算
    v4 := v1 + v2 - v1  // (v1._add(v2))._sub(v1)
    fmt.Println(v4)     // 输出: &{2 3}

    // 自增操作
    v1++                // 调用 v1._inc()
    fmt.Println(v1)     // 输出: &{2 2}
}
```

#### 混合类型与反向运算

MyGo 可以通过反向运算符（*Reverse Operators*）处理混合类型运算。例如，当实现 `Vector` + `Scalar`（`向量` + `标量`）时，如果 `Vector` 未定义对 `Scalar` 的加法，可以由 `Scalar` 定义对 `Vector` 的反向加法。

```go
package main

import "fmt"

type NDArray struct {
	data []int
}

// _add: NDArray + NDArray
func (a *NDArray) _add(b *NDArray) *NDArray {
	// 简化演示：假设长度一致
	res := make([]int, len(a.data))
	for i, v := range a.data {
		res[i] = v + b.data[i]
	}
	return &NDArray{data: res}
}

type Scalar struct {
	val int
}

// _radd: 处理 NDArray + Scalar
// 当左侧是 NDArray 且没有匹配的 _add(Scalar) 时，
// 编译器会尝试调用右侧 Scalar 的 _radd(NDArray)
func (s *Scalar) _radd(arr *NDArray) *NDArray {
	fmt.Println("触发反向运算: Scalar._radd")
	res := make([]int, len(arr.data))
	for i, v := range arr.data {
		res[i] = v + s.val
	}
	return &NDArray{data: res}
}

func main() {
	arr := &NDArray{data: []int{10, 20, 30}}
	num := &Scalar{val: 5}

	// 1. 同类型运算
	// arr._add(arr)
	sumArr := arr + arr
	fmt.Println("Vec + Vec:", sumArr.data)
	// 输出: [20 40 60]

	// 2. 混合类型运算 (触发反向)
	// 流程:
	// a. 查找 arr._add(num) -> 未找到
	// b. 查找 num._radd(arr) -> 找到并调用
	mixed := arr + num
	fmt.Println("Vec + Scalar:", mixed.data)
	// 输出: 触发反向运算: Scalar._radd
	// 输出: [15 25 35]
}
```


### 泛型（Generics）支持说明

MyGO 将操作符重载的特性扩展到了 泛型（Generics） 系统中。这意味着你可以在泛型函数中使用 `+`, `-`, `==`, `[]` 等操作符，只要泛型参数 `T` 满足特定的接口约束。

**核心机制**

在泛型上下文中，编译器无法预知 `T` 具体是什么类型。为了安全地使用操作符，你必须在泛型约束（Interface Constraint）中显式声明对应的“魔法方法”。

- 约束要求：泛型参数 `T` 的约束接口必须包含对应的魔法方法（如 `_add`）。

- 编译重写：当编译器验证 `T` 满足约束后，会自动将标准操作符（如 `a + b`）重写为方法调用（`a._add(b)`）。

#### 示例代码

以下示例展示了如何定义一个支持加法的泛型函数，它既适用于自定义结构体，也适用于原生类型（如 `int`）。

##### 简单示例

```go
package main

import "fmt"

// 1. 定义约束接口
// T 必须实现 _add 方法，且接收参数和返回值均为 T
type Addable[T any] interface {
    _add(T) T
}

// 2. 自定义结构体
type MyInt struct {
    Val int
}

// 实现魔法方法 _add 以支持 + 操作符
func (m MyInt) _add(other MyInt) MyInt {
    return MyInt{Val: m.Val + other.Val}
}

// 3. 泛型函数
// 约束 T 必须满足 Addable[T]
func GenericAdd[T Addable[T]](a, b T) T {
    // 【关键】编译器看到 T 有 _add 约束，
    // 因此允许使用 + 号，并将其重写为 a._add(b)
    return a + b
}

func main() {
    // A. 用于自定义类型
    v1 := MyInt{10}
    v2 := MyInt{20}
    sumObj := GenericAdd(v1, v2)
    fmt.Println(sumObj) // 输出: {30}

    // B. 用于原生类型 (MyGO 特性)
    // 编译器会自动为 int/float 等基础类型合成 _add 方法
    // 因此 int 也满足 Addable[int] 约束
    sumInt := GenericAdd(100, 200)
    fmt.Println(sumInt) // 输出: 300
}
```

##### 范型构造函数

`MyGO` 完美支持泛型类型的构造函数。当使用 `make(GenericType[T], ...)` 时，编译器会根据实例化的具体类型 `T` 来查找和匹配对应的 `_init` 方法。

```go
package main

import (
	"fmt"
)

// 定义泛型结构体
type Box[T any] struct {
	Value T
	Tag   string
}

// 定义泛型构造函数
// 注意：接收者是 *Box[T]
func (b *Box[T]) _init(val T, tag string) {
	b.Value = val
	b.Tag = tag
}

// 支持针对泛型的重载
func (b *Box[T]) _init() {
	// Value 将保持零值
	b.Tag = "default"
}

func main() {
	// 1. 实例化为 int，参数匹配 _init(int, string)
	b1 := make(Box[int], 100, "manual")

	// 2. 实例化为 string，无参匹配 _init()
	b2 := make(Box[string])

	fmt.Printf("b1: %v, %s\n", b1.Value, b1.Tag) // b1: 100, manual
	fmt.Printf("b2: %q, %s\n", b2.Value, b2.Tag) // b2: "", default
}
```

#### 原生类型支持 (Native Types)

MyGO 编译器内置了对原生类型（`int`, `float64`, `string`, `slice`, `map` 等）的方法合成。

即便 `int` 类型在源码中没有定义 `_add` 方法，编译器会在类型检查阶段“伪造”出这些方法。因此，原生类型可以直接满足包含 `_add`, `_sub`, `_getitem` 等方法的接口约束。

注意，对原生类型支持的方式是`零成本抽象`的，也就是说，基本类型会被编译成高效的原生IR(比如`IR.OADD`)，并非使用某些编程语言`装箱`操作！

```go
// 直接使用 slice，它自动满足 _getitem 约束
func GetFirst[T any, S interface{ _getitem(int) T }](seq S) T {
    return seq[0] // 重写为 seq._getitem(0)
}

func main() {
    list := []int{1, 2, 3}
    println(GetFirst(list)) // 输出: 1
}
```

同样，对`构造函数`来说，也是支持的

```go
package main

import "fmt"

type ValIniter[T any] interface {
	_init(pos int)
}

func CreateBoxViaFunc[T ValIniter[T]](val int) *T {
	return make(T, val)
}

type Box[T int] struct {
	Value T
}

func (b *Box[T]) _init(val int) {
	b.Value = T(val)
}

type MySlice []int
type MyMap map[string]int
type MyChan chan int

func main() {
	b := CreateBoxViaFunc[Box[int]](10)
	fmt.Println("box", b.Value)

	s := CreateBoxViaFunc[MySlice](3)
	fmt.Println("slice", len(*s), cap(*s))

	m := CreateBoxViaFunc[MyMap](2)
	(*m)["a"] = 42
	fmt.Println("map", len(*m), (*m)["a"])

	ch := CreateBoxViaFunc[MyChan](2)
	fmt.Println("chan", cap(*ch))
}
```

`slice` 默认"伪实现"了 _init(pos int) _init(pos int, pos cap)

`map/chan` 默认"伪实现"了 _init() _init(pos int)


---

## 注意事项

### 构造函数装饰器的使用

装饰器可以应用到 `_init` 构造函数上，但需要注意以下几点：

**要求**：
- 装饰器函数的签名必须匹配 `_init` 的**原始签名**（未重写前）
- 装饰器必须接受并返回函数类型，并且返回类型要包含 `*TypeName`

**正确示例**：

```go
// 装饰器签名：接受和返回 func(string, int) *Server
func logger(f func(string, int) *Server) func(string, int) *Server {
    return func(host string, port int) *Server {
        fmt.Println("创建服务器:", host, port)
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
    // 输出: 创建服务器: localhost 8080
    fmt.Printf("%s:%d\n", s.host, s.port)
}
```

### 方法重载与默认参数的歧义

当同一个方法既有重载又有默认参数时，可能产生歧义。编译器使用**声明顺序优先**的规则进行匹配。

**示例**：

```go
type DataStore struct {
    intData    map[string]int
    stringData map[string]string
}


// 存储整数
func (ds *DataStore) Set(key string, value int) {
	if ds.intData == nil {
		ds.intData = make(map[string]int)
	}
	ds.intData[key] = value
}

// 存储字符串
func (ds *DataStore) Set(key string, value string) {
	if ds.stringData == nil {
		ds.stringData = make(map[string]string)
	}
	ds.stringData[key] = value
}

// 第一个 Get 方法：处理整数，带默认参数
func (ds *DataStore) Get(key string, defaultValue int = 0) int {
    if v, ok := ds.intData[key]; ok {
        return v
    }
    return defaultValue
}

// 第二个 Get 方法：处理字符串，带默认参数
func (ds *DataStore) Get(key string, defaultValue string = "Unknown") string {
    if v, ok := ds.stringData[key]; ok {
        return v
    }
    return defaultValue
}

func main() {
    store := &DataStore{}
    store.Set("age", 25)
    store.Set("name", "Alice")
    
    // ⚠️ 歧义情况：只传一个参数时
    result := store.Get("someKey")  // 调用第一个方法（int 版本）

    fmt.Println(result)
    // ✅ 明确指定：传入完整参数避免歧义
    intResult := store.Get("age", 0)           // 调用 int 版本
    strResult := store.Get("name", "Unknown")  // 调用 string 版本
	fmt.Println(intResult, strResult)
}

```

**建议**：
- 避免在重载方法中同时使用默认参数
- 如果必须使用，建议显式传递所有参数以避免歧义
- 或者使用不同的方法名（如 `GetInt`、`GetString`）

## 编译和使用

1. 克隆仓库并编译：
```bash
cd src
GOROOT_BOOTSTRAP=/usr/local/go ./make.bash
```

2. 使用自定义的 Go 编译器：
```bash
GOROOT=/path/to/mygo /path/to/mygo/bin/go run your_file.go
```