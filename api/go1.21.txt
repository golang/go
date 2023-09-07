pkg bytes, func ContainsFunc([]uint8, func(int32) bool) bool #54386
pkg bytes, method (*Buffer) AvailableBuffer() []uint8 #53685
pkg bytes, method (*Buffer) Available() int #53685
pkg cmp, func Compare[$0 Ordered]($0, $0) int #59488
pkg cmp, func Less[$0 Ordered]($0, $0) bool #59488
pkg cmp, type Ordered interface {} #59488
pkg context, func AfterFunc(Context, func()) func() bool #57928
pkg context, func WithDeadlineCause(Context, time.Time, error) (Context, CancelFunc) #56661
pkg context, func WithoutCancel(Context) Context #40221
pkg context, func WithTimeoutCause(Context, time.Duration, error) (Context, CancelFunc) #56661
pkg crypto/elliptic, func GenerateKey //deprecated #52221
pkg crypto/elliptic, func Marshal //deprecated #52221
pkg crypto/elliptic, func Unmarshal //deprecated #52221
pkg crypto/elliptic, method (*CurveParams) Add //deprecated #34648
pkg crypto/elliptic, method (*CurveParams) Double //deprecated #34648
pkg crypto/elliptic, method (*CurveParams) IsOnCurve //deprecated #34648
pkg crypto/elliptic, method (*CurveParams) ScalarBaseMult //deprecated #34648
pkg crypto/elliptic, method (*CurveParams) ScalarMult //deprecated #34648
pkg crypto/elliptic, type Curve interface, Add //deprecated #52221
pkg crypto/elliptic, type Curve interface, Double //deprecated #52221
pkg crypto/elliptic, type Curve interface, IsOnCurve //deprecated #52221
pkg crypto/elliptic, type Curve interface, ScalarBaseMult //deprecated #52221
pkg crypto/elliptic, type Curve interface, ScalarMult //deprecated #52221
pkg crypto/rsa, func GenerateMultiPrimeKey //deprecated #56921
pkg crypto/rsa, type PrecomputedValues struct, CRTValues //deprecated #56921
pkg crypto/tls, const QUICEncryptionLevelApplication = 3 #44886
pkg crypto/tls, const QUICEncryptionLevelApplication QUICEncryptionLevel #44886
pkg crypto/tls, const QUICEncryptionLevelEarly = 1 #60107
pkg crypto/tls, const QUICEncryptionLevelEarly QUICEncryptionLevel #60107
pkg crypto/tls, const QUICEncryptionLevelHandshake = 2 #44886
pkg crypto/tls, const QUICEncryptionLevelHandshake QUICEncryptionLevel #44886
pkg crypto/tls, const QUICEncryptionLevelInitial = 0 #44886
pkg crypto/tls, const QUICEncryptionLevelInitial QUICEncryptionLevel #44886
pkg crypto/tls, const QUICHandshakeDone = 7 #44886
pkg crypto/tls, const QUICHandshakeDone QUICEventKind #44886
pkg crypto/tls, const QUICNoEvent = 0 #44886
pkg crypto/tls, const QUICNoEvent QUICEventKind #44886
pkg crypto/tls, const QUICRejectedEarlyData = 6 #60107
pkg crypto/tls, const QUICRejectedEarlyData QUICEventKind #60107
pkg crypto/tls, const QUICSetReadSecret = 1 #44886
pkg crypto/tls, const QUICSetReadSecret QUICEventKind #44886
pkg crypto/tls, const QUICSetWriteSecret = 2 #44886
pkg crypto/tls, const QUICSetWriteSecret QUICEventKind #44886
pkg crypto/tls, const QUICTransportParameters = 4 #44886
pkg crypto/tls, const QUICTransportParameters QUICEventKind #44886
pkg crypto/tls, const QUICTransportParametersRequired = 5 #44886
pkg crypto/tls, const QUICTransportParametersRequired QUICEventKind #44886
pkg crypto/tls, const QUICWriteData = 3 #44886
pkg crypto/tls, const QUICWriteData QUICEventKind #44886
pkg crypto/tls, func NewResumptionState([]uint8, *SessionState) (*ClientSessionState, error) #60105
pkg crypto/tls, func ParseSessionState([]uint8) (*SessionState, error) #60105
pkg crypto/tls, func QUICClient(*QUICConfig) *QUICConn #44886
pkg crypto/tls, func QUICServer(*QUICConfig) *QUICConn #44886
pkg crypto/tls, func VersionName(uint16) string #46308
pkg crypto/tls, method (AlertError) Error() string #44886
pkg crypto/tls, method (*ClientSessionState) ResumptionState() ([]uint8, *SessionState, error) #60105
pkg crypto/tls, method (*Config) DecryptTicket([]uint8, ConnectionState) (*SessionState, error) #60105
pkg crypto/tls, method (*Config) EncryptTicket(ConnectionState, *SessionState) ([]uint8, error) #60105
pkg crypto/tls, method (*QUICConn) Close() error #44886
pkg crypto/tls, method (*QUICConn) ConnectionState() ConnectionState #44886
pkg crypto/tls, method (*QUICConn) HandleData(QUICEncryptionLevel, []uint8) error #44886
pkg crypto/tls, method (*QUICConn) NextEvent() QUICEvent #44886
pkg crypto/tls, method (*QUICConn) SendSessionTicket(QUICSessionTicketOptions) error #60107
pkg crypto/tls, type QUICSessionTicketOptions struct #60107
pkg crypto/tls, type QUICSessionTicketOptions struct, EarlyData bool #60107
pkg crypto/tls, method (*QUICConn) SetTransportParameters([]uint8) #44886
pkg crypto/tls, method (*QUICConn) Start(context.Context) error #44886
pkg crypto/tls, method (QUICEncryptionLevel) String() string #44886
pkg crypto/tls, method (*SessionState) Bytes() ([]uint8, error) #60105
pkg crypto/tls, type AlertError uint8 #44886
pkg crypto/tls, type Config struct, UnwrapSession func([]uint8, ConnectionState) (*SessionState, error) #60105
pkg crypto/tls, type Config struct, WrapSession func(ConnectionState, *SessionState) ([]uint8, error) #60105
pkg crypto/tls, type QUICConfig struct #44886
pkg crypto/tls, type QUICConfig struct, TLSConfig *Config #44886
pkg crypto/tls, type QUICConn struct #44886
pkg crypto/tls, type QUICEncryptionLevel int #44886
pkg crypto/tls, type QUICEventKind int #44886
pkg crypto/tls, type QUICEvent struct #44886
pkg crypto/tls, type QUICEvent struct, Data []uint8 #44886
pkg crypto/tls, type QUICEvent struct, Kind QUICEventKind #44886
pkg crypto/tls, type QUICEvent struct, Level QUICEncryptionLevel #44886
pkg crypto/tls, type QUICEvent struct, Suite uint16 #44886
pkg crypto/tls, type SessionState struct #60105
pkg crypto/tls, type SessionState struct, EarlyData bool #60107
pkg crypto/tls, type SessionState struct, Extra [][]uint8 #60539
pkg crypto/x509, type RevocationListEntry struct #53573
pkg crypto/x509, type RevocationListEntry struct, Extensions []pkix.Extension #53573
pkg crypto/x509, type RevocationListEntry struct, ExtraExtensions []pkix.Extension #53573
pkg crypto/x509, type RevocationListEntry struct, Raw []uint8 #53573
pkg crypto/x509, type RevocationListEntry struct, ReasonCode int #53573
pkg crypto/x509, type RevocationListEntry struct, RevocationTime time.Time #53573
pkg crypto/x509, type RevocationListEntry struct, SerialNumber *big.Int #53573
pkg crypto/x509, type RevocationList struct, RevokedCertificateEntries []RevocationListEntry #53573
pkg crypto/x509, type RevocationList struct, RevokedCertificates //deprecated #53573
pkg debug/elf, const COMPRESS_ZSTD = 2 #55107
pkg debug/elf, const COMPRESS_ZSTD CompressionType #55107
pkg debug/elf, const DF_1_CONFALT = 8192 #56887
pkg debug/elf, const DF_1_CONFALT DynFlag1 #56887
pkg debug/elf, const DF_1_DIRECT = 256 #56887
pkg debug/elf, const DF_1_DIRECT DynFlag1 #56887
pkg debug/elf, const DF_1_DISPRELDNE = 32768 #56887
pkg debug/elf, const DF_1_DISPRELDNE DynFlag1 #56887
pkg debug/elf, const DF_1_DISPRELPND = 65536 #56887
pkg debug/elf, const DF_1_DISPRELPND DynFlag1 #56887
pkg debug/elf, const DF_1_EDITED = 2097152 #56887
pkg debug/elf, const DF_1_EDITED DynFlag1 #56887
pkg debug/elf, const DF_1_ENDFILTEE = 16384 #56887
pkg debug/elf, const DF_1_ENDFILTEE DynFlag1 #56887
pkg debug/elf, const DF_1_GLOBAL = 2 #56887
pkg debug/elf, const DF_1_GLOBAL DynFlag1 #56887
pkg debug/elf, const DF_1_GLOBAUDIT = 16777216 #56887
pkg debug/elf, const DF_1_GLOBAUDIT DynFlag1 #56887
pkg debug/elf, const DF_1_GROUP = 4 #56887
pkg debug/elf, const DF_1_GROUP DynFlag1 #56887
pkg debug/elf, const DF_1_IGNMULDEF = 262144 #56887
pkg debug/elf, const DF_1_IGNMULDEF DynFlag1 #56887
pkg debug/elf, const DF_1_INITFIRST = 32 #56887
pkg debug/elf, const DF_1_INITFIRST DynFlag1 #56887
pkg debug/elf, const DF_1_INTERPOSE = 1024 #56887
pkg debug/elf, const DF_1_INTERPOSE DynFlag1 #56887
pkg debug/elf, const DF_1_KMOD = 268435456 #56887
pkg debug/elf, const DF_1_KMOD DynFlag1 #56887
pkg debug/elf, const DF_1_LOADFLTR = 16 #56887
pkg debug/elf, const DF_1_LOADFLTR DynFlag1 #56887
pkg debug/elf, const DF_1_NOCOMMON = 1073741824 #56887
pkg debug/elf, const DF_1_NOCOMMON DynFlag1 #56887
pkg debug/elf, const DF_1_NODEFLIB = 2048 #56887
pkg debug/elf, const DF_1_NODEFLIB DynFlag1 #56887
pkg debug/elf, const DF_1_NODELETE = 8 #56887
pkg debug/elf, const DF_1_NODELETE DynFlag1 #56887
pkg debug/elf, const DF_1_NODIRECT = 131072 #56887
pkg debug/elf, const DF_1_NODIRECT DynFlag1 #56887
pkg debug/elf, const DF_1_NODUMP = 4096 #56887
pkg debug/elf, const DF_1_NODUMP DynFlag1 #56887
pkg debug/elf, const DF_1_NOHDR = 1048576 #56887
pkg debug/elf, const DF_1_NOHDR DynFlag1 #56887
pkg debug/elf, const DF_1_NOKSYMS = 524288 #56887
pkg debug/elf, const DF_1_NOKSYMS DynFlag1 #56887
pkg debug/elf, const DF_1_NOOPEN = 64 #56887
pkg debug/elf, const DF_1_NOOPEN DynFlag1 #56887
pkg debug/elf, const DF_1_NORELOC = 4194304 #56887
pkg debug/elf, const DF_1_NORELOC DynFlag1 #56887
pkg debug/elf, const DF_1_NOW = 1 #56887
pkg debug/elf, const DF_1_NOW DynFlag1 #56887
pkg debug/elf, const DF_1_ORIGIN = 128 #56887
pkg debug/elf, const DF_1_ORIGIN DynFlag1 #56887
pkg debug/elf, const DF_1_PIE = 134217728 #56887
pkg debug/elf, const DF_1_PIE DynFlag1 #56887
pkg debug/elf, const DF_1_SINGLETON = 33554432 #56887
pkg debug/elf, const DF_1_SINGLETON DynFlag1 #56887
pkg debug/elf, const DF_1_STUB = 67108864 #56887
pkg debug/elf, const DF_1_STUB DynFlag1 #56887
pkg debug/elf, const DF_1_SYMINTPOSE = 8388608 #56887
pkg debug/elf, const DF_1_SYMINTPOSE DynFlag1 #56887
pkg debug/elf, const DF_1_TRANS = 512 #56887
pkg debug/elf, const DF_1_TRANS DynFlag1 #56887
pkg debug/elf, const DF_1_WEAKFILTER = 536870912 #56887
pkg debug/elf, const DF_1_WEAKFILTER DynFlag1 #56887
pkg debug/elf, const R_PPC64_REL24_P9NOTOC = 124 #60348
pkg debug/elf, const R_PPC64_REL24_P9NOTOC R_PPC64 #60348
pkg debug/elf, method (DynFlag1) GoString() string #56887
pkg debug/elf, method (DynFlag1) String() string #56887
pkg debug/elf, method (*File) DynValue(DynTag) ([]uint64, error) #56892
pkg debug/elf, type DynFlag1 uint32 #56887
pkg encoding/binary, var NativeEndian nativeEndian #57237
pkg errors, var ErrUnsupported error #41198
pkg flag, func BoolFunc(string, string, func(string) error) #53747
pkg flag, method (*FlagSet) BoolFunc(string, string, func(string) error) #53747
pkg go/ast, func IsGenerated(*File) bool #28089
pkg go/ast, func NewPackage //deprecated #52463
pkg go/ast, type File struct, GoVersion string #59033
pkg go/ast, type Importer //deprecated #52463
pkg go/ast, type Object //deprecated #52463
pkg go/ast, type Package //deprecated #52463
pkg go/ast, type Scope //deprecated #52463
pkg go/build/constraint, func GoVersion(Expr) string #59033
pkg go/build, type Directive struct #56986
pkg go/build, type Directive struct, Pos token.Position #56986
pkg go/build, type Directive struct, Text string #56986
pkg go/build, type Package struct, Directives []Directive #56986
pkg go/build, type Package struct, TestDirectives []Directive #56986
pkg go/build, type Package struct, XTestDirectives []Directive #56986
pkg go/token, method (*File) Lines() []int #57708
pkg go/types, method (*Package) GoVersion() string #61175
pkg html/template, const ErrJSTemplate = 12 #59584
pkg html/template, const ErrJSTemplate ErrorCode #59584
pkg io/fs, func FormatDirEntry(DirEntry) string #54451
pkg io/fs, func FormatFileInfo(FileInfo) string #54451
pkg log/slog, const KindAny = 0 #56345
pkg log/slog, const KindAny Kind #56345
pkg log/slog, const KindBool = 1 #56345
pkg log/slog, const KindBool Kind #56345
pkg log/slog, const KindDuration = 2 #56345
pkg log/slog, const KindDuration Kind #56345
pkg log/slog, const KindFloat64 = 3 #56345
pkg log/slog, const KindFloat64 Kind #56345
pkg log/slog, const KindGroup = 8 #56345
pkg log/slog, const KindGroup Kind #56345
pkg log/slog, const KindInt64 = 4 #56345
pkg log/slog, const KindInt64 Kind #56345
pkg log/slog, const KindLogValuer = 9 #56345
pkg log/slog, const KindLogValuer Kind #56345
pkg log/slog, const KindString = 5 #56345
pkg log/slog, const KindString Kind #56345
pkg log/slog, const KindTime = 6 #56345
pkg log/slog, const KindTime Kind #56345
pkg log/slog, const KindUint64 = 7 #56345
pkg log/slog, const KindUint64 Kind #56345
pkg log/slog, const LevelDebug = -4 #56345
pkg log/slog, const LevelDebug Level #56345
pkg log/slog, const LevelError = 8 #56345
pkg log/slog, const LevelError Level #56345
pkg log/slog, const LevelInfo = 0 #56345
pkg log/slog, const LevelInfo Level #56345
pkg log/slog, const LevelKey ideal-string #56345
pkg log/slog, const LevelKey = "level" #56345
pkg log/slog, const LevelWarn = 4 #56345
pkg log/slog, const LevelWarn Level #56345
pkg log/slog, const MessageKey ideal-string #56345
pkg log/slog, const MessageKey = "msg" #56345
pkg log/slog, const SourceKey ideal-string #56345
pkg log/slog, const SourceKey = "source" #56345
pkg log/slog, const TimeKey ideal-string #56345
pkg log/slog, const TimeKey = "time" #56345
pkg log/slog, func Any(string, interface{}) Attr #56345
pkg log/slog, func AnyValue(interface{}) Value #56345
pkg log/slog, func Bool(string, bool) Attr #56345
pkg log/slog, func BoolValue(bool) Value #56345
pkg log/slog, func DebugContext(context.Context, string, ...interface{}) #61200
pkg log/slog, func Debug(string, ...interface{}) #56345
pkg log/slog, func Default() *Logger #56345
pkg log/slog, func Duration(string, time.Duration) Attr #56345
pkg log/slog, func DurationValue(time.Duration) Value #56345
pkg log/slog, func ErrorContext(context.Context, string, ...interface{}) #61200
pkg log/slog, func Error(string, ...interface{}) #56345
pkg log/slog, func Float64(string, float64) Attr #56345
pkg log/slog, func Float64Value(float64) Value #56345
pkg log/slog, func Group(string, ...interface{}) Attr #59204
pkg log/slog, func GroupValue(...Attr) Value #56345
pkg log/slog, func InfoContext(context.Context, string, ...interface{}) #61200
pkg log/slog, func Info(string, ...interface{}) #56345
pkg log/slog, func Int64(string, int64) Attr #56345
pkg log/slog, func Int64Value(int64) Value #56345
pkg log/slog, func Int(string, int) Attr #56345
pkg log/slog, func IntValue(int) Value #56345
pkg log/slog, func LogAttrs(context.Context, Level, string, ...Attr) #56345
pkg log/slog, func Log(context.Context, Level, string, ...interface{}) #56345
pkg log/slog, func New(Handler) *Logger #56345
pkg log/slog, func NewJSONHandler(io.Writer, *HandlerOptions) *JSONHandler #59339
pkg log/slog, func NewLogLogger(Handler, Level) *log.Logger #56345
pkg log/slog, func NewRecord(time.Time, Level, string, uintptr) Record #56345
pkg log/slog, func NewTextHandler(io.Writer, *HandlerOptions) *TextHandler #59339
pkg log/slog, func SetDefault(*Logger) #56345
pkg log/slog, func String(string, string) Attr #56345
pkg log/slog, func StringValue(string) Value #56345
pkg log/slog, func Time(string, time.Time) Attr #56345
pkg log/slog, func TimeValue(time.Time) Value #56345
pkg log/slog, func Uint64(string, uint64) Attr #56345
pkg log/slog, func Uint64Value(uint64) Value #56345
pkg log/slog, func WarnContext(context.Context, string, ...interface{})  #61200
pkg log/slog, func Warn(string, ...interface{}) #56345
pkg log/slog, func With(...interface{}) *Logger #56345
pkg log/slog, method (Attr) Equal(Attr) bool #56345
pkg log/slog, method (Attr) String() string #56345
pkg log/slog, method (*JSONHandler) Enabled(context.Context, Level) bool #56345
pkg log/slog, method (*JSONHandler) Handle(context.Context, Record) error #56345
pkg log/slog, method (*JSONHandler) WithAttrs([]Attr) Handler #56345
pkg log/slog, method (*JSONHandler) WithGroup(string) Handler #56345
pkg log/slog, method (Kind) String() string #56345
pkg log/slog, method (Level) Level() Level #56345
pkg log/slog, method (Level) MarshalJSON() ([]uint8, error) #56345
pkg log/slog, method (Level) MarshalText() ([]uint8, error) #56345
pkg log/slog, method (Level) String() string #56345
pkg log/slog, method (*Level) UnmarshalJSON([]uint8) error #56345
pkg log/slog, method (*Level) UnmarshalText([]uint8) error #56345
pkg log/slog, method (*LevelVar) Level() Level #56345
pkg log/slog, method (*LevelVar) MarshalText() ([]uint8, error) #56345
pkg log/slog, method (*LevelVar) Set(Level) #56345
pkg log/slog, method (*LevelVar) String() string #56345
pkg log/slog, method (*LevelVar) UnmarshalText([]uint8) error #56345
pkg log/slog, method (*Logger) DebugContext(context.Context, string, ...interface{}) #61200
pkg log/slog, method (*Logger) Debug(string, ...interface{}) #56345
pkg log/slog, method (*Logger) Enabled(context.Context, Level) bool #56345
pkg log/slog, method (*Logger) ErrorContext(context.Context, string, ...interface{}) #61200
pkg log/slog, method (*Logger) Error(string, ...interface{}) #56345
pkg log/slog, method (*Logger) Handler() Handler #56345
pkg log/slog, method (*Logger) InfoContext(context.Context, string, ...interface{}) #61200
pkg log/slog, method (*Logger) Info(string, ...interface{}) #56345
pkg log/slog, method (*Logger) LogAttrs(context.Context, Level, string, ...Attr) #56345
pkg log/slog, method (*Logger) Log(context.Context, Level, string, ...interface{}) #56345
pkg log/slog, method (*Logger) WarnContext(context.Context, string, ...interface{}) #61200
pkg log/slog, method (*Logger) Warn(string, ...interface{}) #56345
pkg log/slog, method (*Logger) WithGroup(string) *Logger #56345
pkg log/slog, method (*Logger) With(...interface{}) *Logger #56345
pkg log/slog, method (*Record) AddAttrs(...Attr) #56345
pkg log/slog, method (*Record) Add(...interface{}) #56345
pkg log/slog, method (Record) Attrs(func(Attr) bool) #59060
pkg log/slog, method (Record) Clone() Record #56345
pkg log/slog, method (Record) NumAttrs() int #56345
pkg log/slog, method (*TextHandler) Enabled(context.Context, Level) bool #56345
pkg log/slog, method (*TextHandler) Handle(context.Context, Record) error #56345
pkg log/slog, method (*TextHandler) WithAttrs([]Attr) Handler #56345
pkg log/slog, method (*TextHandler) WithGroup(string) Handler #56345
pkg log/slog, method (Value) Any() interface{} #56345
pkg log/slog, method (Value) Bool() bool #56345
pkg log/slog, method (Value) Duration() time.Duration #56345
pkg log/slog, method (Value) Equal(Value) bool #56345
pkg log/slog, method (Value) Float64() float64 #56345
pkg log/slog, method (Value) Group() []Attr #56345
pkg log/slog, method (Value) Int64() int64 #56345
pkg log/slog, method (Value) Kind() Kind #56345
pkg log/slog, method (Value) LogValuer() LogValuer #56345
pkg log/slog, method (Value) Resolve() Value #56345
pkg log/slog, method (Value) String() string #56345
pkg log/slog, method (Value) Time() time.Time #56345
pkg log/slog, method (Value) Uint64() uint64 #56345
pkg log/slog, type Attr struct #56345
pkg log/slog, type Attr struct, Key string #56345
pkg log/slog, type Attr struct, Value Value #56345
pkg log/slog, type Handler interface, Enabled(context.Context, Level) bool #56345
pkg log/slog, type Handler interface { Enabled, Handle, WithAttrs, WithGroup } #56345
pkg log/slog, type Handler interface, Handle(context.Context, Record) error #56345
pkg log/slog, type Handler interface, WithAttrs([]Attr) Handler #56345
pkg log/slog, type Handler interface, WithGroup(string) Handler #56345
pkg log/slog, type HandlerOptions struct #56345
pkg log/slog, type HandlerOptions struct, AddSource bool #56345
pkg log/slog, type HandlerOptions struct, Level Leveler #56345
pkg log/slog, type HandlerOptions struct, ReplaceAttr func([]string, Attr) Attr #56345
pkg log/slog, type JSONHandler struct #56345
pkg log/slog, type Kind int #56345
pkg log/slog, type Leveler interface { Level } #56345
pkg log/slog, type Leveler interface, Level() Level #56345
pkg log/slog, type Level int #56345
pkg log/slog, type LevelVar struct #56345
pkg log/slog, type Logger struct #56345
pkg log/slog, type LogValuer interface { LogValue } #56345
pkg log/slog, type LogValuer interface, LogValue() Value #56345
pkg log/slog, type Record struct #56345
pkg log/slog, type Record struct, Level Level #56345
pkg log/slog, type Record struct, Message string #56345
pkg log/slog, type Record struct, PC uintptr #56345
pkg log/slog, type Record struct, Time time.Time #56345
pkg log/slog, type Source struct #59280
pkg log/slog, type Source struct, File string #59280
pkg log/slog, type Source struct, Function string #59280
pkg log/slog, type Source struct, Line int #59280
pkg log/slog, type TextHandler struct #56345
pkg log/slog, type Value struct #56345
pkg maps, func Clone[$0 interface{ ~map[$1]$2 }, $1 comparable, $2 interface{}]($0) $0 #57436
pkg maps, func Copy[$0 interface{ ~map[$2]$3 }, $1 interface{ ~map[$2]$3 }, $2 comparable, $3 interface{}]($0, $1) #57436
pkg maps, func DeleteFunc[$0 interface{ ~map[$1]$2 }, $1 comparable, $2 interface{}]($0, func($1, $2) bool) #57436
pkg maps, func Equal[$0 interface{ ~map[$2]$3 }, $1 interface{ ~map[$2]$3 }, $2 comparable, $3 comparable]($0, $1) bool #57436
pkg maps, func EqualFunc[$0 interface{ ~map[$2]$3 }, $1 interface{ ~map[$2]$4 }, $2 comparable, $3 interface{}, $4 interface{}]($0, $1, func($3, $4) bool) bool #57436
pkg math/big, method (*Int) Float64() (float64, Accuracy) #56984
pkg net/http, method (*ProtocolError) Is(error) bool #41198
pkg net/http, method (*ResponseController) EnableFullDuplex() error #57786
pkg net/http, var ErrSchemeMismatch error #44855
pkg net, method (*Dialer) MultipathTCP() bool #56539
pkg net, method (*Dialer) SetMultipathTCP(bool) #56539
pkg net, method (*ListenConfig) MultipathTCP() bool #56539
pkg net, method (*ListenConfig) SetMultipathTCP(bool) #56539
pkg net, method (*TCPConn) MultipathTCP() (bool, error) #59166
pkg reflect, method (Value) Clear() #55002
pkg reflect, type SliceHeader //deprecated #56906
pkg reflect, type StringHeader //deprecated #56906
pkg regexp, method (*Regexp) MarshalText() ([]uint8, error) #46159
pkg regexp, method (*Regexp) UnmarshalText([]uint8) error #46159
pkg runtime, method (*PanicNilError) Error() string #25448
pkg runtime, method (*PanicNilError) RuntimeError() #25448
pkg runtime, method (*Pinner) Pin(interface{}) #46787
pkg runtime, method (*Pinner) Unpin() #46787
pkg runtime, type PanicNilError struct #25448
pkg runtime, type Pinner struct #46787
pkg slices, func BinarySearch[$0 interface{ ~[]$1 }, $1 cmp.Ordered]($0, $1) (int, bool) #60091
pkg slices, func BinarySearchFunc[$0 interface{ ~[]$1 }, $1 interface{}, $2 interface{}]($0, $2, func($1, $2) int) (int, bool) #60091
pkg slices, func Clip[$0 interface{ ~[]$1 }, $1 interface{}]($0) $0 #57433
pkg slices, func Clone[$0 interface{ ~[]$1 }, $1 interface{}]($0) $0 #57433
pkg slices, func Compact[$0 interface{ ~[]$1 }, $1 comparable]($0) $0 #57433
pkg slices, func CompactFunc[$0 interface{ ~[]$1 }, $1 interface{}]($0, func($1, $1) bool) $0 #57433
pkg slices, func Compare[$0 interface{ ~[]$1 }, $1 cmp.Ordered]($0, $0) int #60091
pkg slices, func CompareFunc[$0 interface{ ~[]$2 }, $1 interface{ ~[]$3 }, $2 interface{}, $3 interface{}]($0, $1, func($2, $3) int) int #60091
pkg slices, func Contains[$0 interface{ ~[]$1 }, $1 comparable]($0, $1) bool #57433
pkg slices, func ContainsFunc[$0 interface{ ~[]$1 }, $1 interface{}]($0, func($1) bool) bool #57433
pkg slices, func Delete[$0 interface{ ~[]$1 }, $1 interface{}]($0, int, int) $0 #57433
pkg slices, func DeleteFunc[$0 interface{ ~[]$1 }, $1 interface{}]($0, func($1) bool) $0 #54768
pkg slices, func Equal[$0 interface{ ~[]$1 }, $1 comparable]($0, $0) bool #57433
pkg slices, func EqualFunc[$0 interface{ ~[]$2 }, $1 interface{ ~[]$3 }, $2 interface{}, $3 interface{}]($0, $1, func($2, $3) bool) bool #57433
pkg slices, func Grow[$0 interface{ ~[]$1 }, $1 interface{}]($0, int) $0 #57433
pkg slices, func Index[$0 interface{ ~[]$1 }, $1 comparable]($0, $1) int #57433
pkg slices, func IndexFunc[$0 interface{ ~[]$1 }, $1 interface{}]($0, func($1) bool) int #57433
pkg slices, func Insert[$0 interface{ ~[]$1 }, $1 interface{}]($0, int, ...$1) $0 #57433
pkg slices, func IsSorted[$0 interface{ ~[]$1 }, $1 cmp.Ordered]($0) bool #60091
pkg slices, func IsSortedFunc[$0 interface{ ~[]$1 }, $1 interface{}]($0, func($1, $1) int) bool #60091
pkg slices, func Max[$0 interface{ ~[]$1 }, $1 cmp.Ordered]($0) $1 #60091
pkg slices, func MaxFunc[$0 interface{ ~[]$1 }, $1 interface{}]($0, func($1, $1) int) $1 #60091
pkg slices, func Min[$0 interface{ ~[]$1 }, $1 cmp.Ordered]($0) $1 #60091
pkg slices, func MinFunc[$0 interface{ ~[]$1 }, $1 interface{}]($0, func($1, $1) int) $1 #60091
pkg slices, func Replace[$0 interface{ ~[]$1 }, $1 interface{}]($0, int, int, ...$1) $0 #57433
pkg slices, func Reverse[$0 interface{ ~[]$1 }, $1 interface{}]($0) #58565
pkg slices, func Sort[$0 interface{ ~[]$1 }, $1 cmp.Ordered]($0) #60091
pkg slices, func SortFunc[$0 interface{ ~[]$1 }, $1 interface{}]($0, func($1, $1) int) #60091
pkg slices, func SortStableFunc[$0 interface{ ~[]$1 }, $1 interface{}]($0, func($1, $1) int) #60091
pkg strings, func ContainsFunc(string, func(int32) bool) bool #54386
pkg sync, func OnceFunc(func()) func() #56102
pkg sync, func OnceValue[$0 interface{}](func() $0) func() $0 #56102
pkg sync, func OnceValues[$0 interface{}, $1 interface{}](func() ($0, $1)) func() ($0, $1) #56102
pkg syscall (freebsd-386-cgo), type SysProcAttr struct, Jail int #46259
pkg syscall (freebsd-386), type SysProcAttr struct, Jail int #46259
pkg syscall (freebsd-amd64-cgo), type SysProcAttr struct, Jail int #46259
pkg syscall (freebsd-amd64), type SysProcAttr struct, Jail int #46259
pkg syscall (freebsd-arm64-cgo), type SysProcAttr struct, Jail int #46259
pkg syscall (freebsd-arm64), type SysProcAttr struct, Jail int #46259
pkg syscall (freebsd-arm-cgo), type SysProcAttr struct, Jail int #46259
pkg syscall (freebsd-arm), type SysProcAttr struct, Jail int #46259
pkg syscall (freebsd-riscv64-cgo), type SysProcAttr struct, Jail int #46259
pkg syscall (freebsd-riscv64), type SysProcAttr struct, Jail int #46259
pkg testing, func Testing() bool #52600
pkg testing/slogtest, func TestHandler(slog.Handler, func() []map[string]interface{}) error #56345
pkg unicode, const Version = "15.0.0" #55079
pkg unicode, var Cypro_Minoan *RangeTable #55079
pkg unicode, var Kawi *RangeTable #55079
pkg unicode, var Nag_Mundari *RangeTable #55079
pkg unicode, var Old_Uyghur *RangeTable #55079
pkg unicode, var Tangsa *RangeTable #55079
pkg unicode, var Toto *RangeTable #55079
pkg unicode, var Vithkuqi *RangeTable #55079
