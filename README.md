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

*重要! 需要导入`reflect`包！*

```go
import "reflect" // 重要！目前只能通过reflect实现?.表达式！

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
    
    fmt.Println(email1)  // alice@example.com
    fmt.Println(email2)  // <nil>
}
```

## 4. 三元表达式

支持简洁的条件表达式 `condition ? trueValue : falseValue`。

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
    
    // 使用三元表达式
    status = age >= 18 ? "成年" : "未成年"
    fmt.Println(status)  // 输出: 成年
    
    // 短格式（省略 else，默认使用 true 值）
    result := age >= 18 ? "已成年"
    fmt.Println(result)  // 输出: 已成年
    
    // 嵌套三元表达式
    score := 85
    grade := score >= 90 ? "A" : score >= 80 ? "B" : score >= 60 ? "C" : "D"
    fmt.Println(grade)   // 输出: B
}
```

---

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