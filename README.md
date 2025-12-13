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