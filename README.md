# It's MyGO!!!!!

本仓库为GO(fork)的非官方个人改造版本，因此叫`MyGO`.

![MyGO image](img/mygo1.png)
*MyGO image by Gemini/Nano-banana-pro*

![MyGO Logo](img/mygo2.jpg)
*MyGO的Logo*

--------------

本仓库已经实现的特性有以下四个

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

可以在同一个函数上应用多个装饰器，按从下到上的顺序执行。

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

## 6. 构造函数

使用 `make(TypeName, args...)` 语法创建结构体实例，支持自定义初始化逻辑。

**规则**：
- 构造方法必须命名为 `_init`
- 必须是指针接收器方法（`func (t *Type) _init(...)`）
- 不需要手动写返回值（编译器自动添加）
- 支持重载（不同参数类型）和默认参数

### 6.1 基础用法

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

### 6.2 构造函数重载

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

### 6.3 构造函数 + 默认参数

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

## 7. 魔法函数(实验特性)

注意，标记为实验特性的可能有BUG，慎用 (魔法方法所有的特性主要为未来的向量计算作准备)

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

// _setitem: 支持 person["name"] = value 语法
func (p *Person) _setitem(name string, value string) {
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