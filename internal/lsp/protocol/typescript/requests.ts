import * as fs from 'fs';
import * as ts from 'typescript';

// generate tsclient.go and tsserver.go, which are the definitions and stubs for
// supporting the LPS protocol. These files have 3 sections:
// 1. define the Client or Server type
// 2. fill out the clientHandler or serveHandler which is basically a large
//    switch on the requests and notifications received by the client/server.
// 3. The methods corresponding to these. (basically parse the request,
//    call something, and perhaps send a response.)

let dir = process.env['HOME'];
let fnames = [
  `/vscode-languageserver-node/protocol/src/protocol.ts`,
  `/vscode-languageserver-node/jsonrpc/src/main.ts`
];

let fda: number, fdy: number;  // file descriptors

function createOutputFiles() {
  fda = fs.openSync('/tmp/ts-a', 'w')  // dump of AST
  fdy = fs.openSync('/tmp/ts-c', 'w')  // unused, for debugging
}
function pra(s: string) {
  return (fs.writeSync(fda, s))
}
function prb(s: string) {
  return (fs.writeSync(fdy, s + '\n'))
}

let program: ts.Program;

function generate(files: string[], options: ts.CompilerOptions): void {
  program = ts.createProgram(files, options);
  program.getTypeChecker();

  dumpAST();  // for debugging

  // visit every sourceFile in the program, collecting information
  for (const sourceFile of program.getSourceFiles()) {
    if (!sourceFile.isDeclarationFile) {
      ts.forEachChild(sourceFile, genStuff)
    }
  }
  // when 4 args, they are param, result, error, registration options, e.g.:
  // RequestType<TextDocumentPositionParams, Definition | DefinitionLink[] |
  // null,
  //   void, TextDocumentRegistrationOptions>('textDocument/implementation');
  // 3 args is RequestType0('shutdown')<void, void, void>
  // and RequestType0('workspace/workspaceFolders)<WorkspaceFolder[]|null, void,
  // void>

  // the two args are the notification data and the registration data
  // except for textDocument/selectionRange and a NotificationType0('exit')
  // selectionRange is the following, but for now do it by hand, special case.
  // RequestType<TextDocumentPositionParams, SelectionRange[] | null, any, any>
  //    = new RequestType('textDocument/selectionRange')
  // and foldingRange has the same problem.

  setReceives();  // distinguish client and server
  // for each of Client and Server there are 3 parts to the output:
  // 1. type X interface {methods}
  // 2. func (h *serverHandler) Deliver(...) { switch r.method }
  // 3. func (x *xDispatcher) Method(ctx, parm)
  not.forEach(
      (v, k) => {receives.get(k) == 'client' ? goNot(client, k) :
                                               goNot(server, k)});
  req.forEach(
      (v, k) => {receives.get(k) == 'client' ? goReq(client, k) :
                                               goReq(server, k)});
  // and print the Go code
  output(client);
  output(server);
  return;
}

// Go signatures for methods.
function sig(nm: string, a: string, b: string, names?: boolean): string {
  if (a != '') {
    if (names)
      a = ', params *' + a;
    else
      a = ', *' + a;
  }
  let ret = 'error';
  if (b != '') {
    b.startsWith('[]') || b.startsWith('interface') || (b = '*' + b);
    ret = `(${b}, error)`;
  }
  let start = `${nm}(`;
  if (names) {
    start = start + 'ctx ';
  }
  return `${start}context.Context${a}) ${ret}`;
}

const notNil = `if r.Params != nil {
				r.Reply(ctx, nil, jsonrpc2.NewErrorf(jsonrpc2.CodeInvalidParams, "Expected no params"))
				return true
			}`;
// Go code for notifications. Side is client or server, m is the request method
function goNot(side: side, m: string) {
  const n = not.get(m);
  let a = goType(side, m, n.typeArguments[0]);
  // let b = goType(m, n.typeArguments[1]); These are registration options
  const nm = methodName(m);
  side.methods.push(sig(nm, a, ''));
  const caseHdr = `case "${m}": // notif`;
  let case1 = notNil;
  if (a != '') {
    case1 = `var params ${a}
    if err := json.Unmarshal(*r.Params, &params); err != nil {
      sendParseError(ctx, r, err)
      return true
    }
    if err := h.${side.name}.${nm}(ctx, &params); err != nil {
      log.Error(ctx, "", err)
    }
    return true`;
  } else {
    case1 = `if err := h.${side.name}.${nm}(ctx); err != nil {
      log.Error(ctx, "", err)
    }
    return true`;
  }
  side.cases.push(`${caseHdr}\n${case1}`);

  const arg3 = a == '' ? 'nil' : 'params';
  side.calls.push(`
  func (s *${side.name}Dispatcher) ${sig(nm, a, '', true)} {
    return s.Conn.Notify(ctx, "${m}", ${arg3})
  }`);
}

// Go code for requests.
function goReq(side: side, m: string) {
  const n = req.get(m);

  const nm = methodName(m);
  let a = goType(side, m, n.typeArguments[0]);
  let b = goType(side, m, n.typeArguments[1]);
  if (n.getText().includes('Type0')) {
    b = a;
    a = '';  // workspace/workspaceFolders and shutdown
  }
  prb(`${side.name} req ${a != ''},${b != ''} ${nm} ${m} ${loc(n)}`)
  side.methods.push(sig(nm, a, b));

  const caseHdr = `case "${m}": // req`;
  let case1 = notNil;
  if (a != '') {
    case1 = `var params ${a}
    if err := json.Unmarshal(*r.Params, &params); err != nil {
      sendParseError(ctx, r, err)
      return true
    }`;
  }
  const arg2 = a == '' ? '' : ', &params';
  let case2 = `if err := h.${side.name}.${nm}(ctx${arg2}); err != nil {
    log.Error(ctx, "", err)
  }`;
  if (b != '') {
    case2 = `resp, err := h.${side.name}.${nm}(ctx${arg2})
    if err := r.Reply(ctx, resp, err); err != nil {
      log.Error(ctx, "", err)
    }
    return true`;
  } else {  // response is nil
    case2 = `err := h.${side.name}.${nm}(ctx${arg2})
    if err := r.Reply(ctx, nil, err); err != nil {
      log.Error(ctx, "", err)
    }
    return true`
  }

  side.cases.push(`${caseHdr}\n${case1}\n${case2}`);

  const callHdr = `func (s *${side.name}Dispatcher) ${sig(nm, a, b, true)} {`;
  let callBody = `return s.Conn.Call(ctx, "${m}", nil, nil)\n}`;
  if (b != '') {
    const p2 = a == '' ? 'nil' : 'params';
    let theRet = `result`;
    !b.startsWith('[]') && !b.startsWith('interface') && (theRet = '&result');
    callBody = `var result ${b}
			if err := s.Conn.Call(ctx, "${m}", ${
        p2}, &result); err != nil {
				return nil, err
      }
      return ${theRet}, nil
    }`;
  } else if (a != '') {
    callBody = `return s.Conn.Call(ctx, "${m}", params, nil) // Call, not Notify
  }`
  }
  side.calls.push(`${callHdr}\n${callBody}\n`);
}

// make sure method names are unique
let seenNames = new Set<string>();
function methodName(m: string): string {
  const i = m.indexOf('/');
  let s = m.substring(i + 1);
  let x = s[0].toUpperCase() + s.substring(1);
  if (seenNames.has(x)) {
    x += m[0].toUpperCase() + m.substring(1, i);
  }
  seenNames.add(x);
  return x;
}

function output(side: side) {
  if (side.outputFile === undefined) side.outputFile = `ts${side.name}.go`;
  side.fd = fs.openSync(side.outputFile, 'w');
  const f = function(s: string) {
    fs.writeSync(side.fd, s);
    fs.writeSync(side.fd, '\n');
  };
  f(`package protocol`);
  f(`// Code generated (see typescript/README.md) DO NOT EDIT.\n`);
  f(`
  import (
    "context"
    "encoding/json"

    "golang.org/x/tools/internal/jsonrpc2"
    "golang.org/x/tools/internal/telemetry/log"
  )
  `);
  const a = side.name[0].toUpperCase() + side.name.substring(1)
  f(`type ${a} interface {`);
  side.methods.forEach((v) => {f(v)});
  f('}\n');
  f(`func (h ${
      side.name}Handler) Deliver(ctx context.Context, r *jsonrpc2.Request, delivered bool) bool {
      if delivered {
        return false
      }
      switch r.Method {
      case "$/cancelRequest":
        var params CancelParams
        if err := json.Unmarshal(*r.Params, &params); err != nil {
          sendParseError(ctx, r, err)
          return true
        }
        r.Conn().Cancel(params.ID)
        return true`);
  side.cases.forEach((v) => {f(v)});
  f(`
  default:
    return false
  }
}`);
  f(`
  type ${side.name}Dispatcher struct {
    *jsonrpc2.Conn
  }
  `);
  side.calls.forEach((v) => {f(v)});
  if (side.name == 'server')
    f(`
  type CancelParams struct {
    /**
     * The request id to cancel.
     */
    ID jsonrpc2.ID \`json:"id"\`
  }`);
  f(`// Types constructed to avoid structs as formal argument types`)
  side.ourTypes.forEach((val, key) => f(`type ${val} ${key}`));
}

interface side {
  methods: string[];
  cases: string[];
  calls: string[];
  ourTypes: Map<string, string>;
  name: string;    // client or server
  goName: string;  // Client or Server
  outputFile?: string;
  fd?: number
}
let client: side = {
  methods: [],
  cases: [],
  calls: [],
  name: 'client',
  goName: 'Client',
  ourTypes: new Map<string, string>()
};
let server: side = {
  methods: [],
  cases: [],
  calls: [],
  name: 'server',
  goName: 'Server',
  ourTypes: new Map<string, string>()
};

let req = new Map<string, ts.NewExpression>();        // requests
let not = new Map<string, ts.NewExpression>();        // notifications
let receives = new Map<string, 'server'|'client'>();  // who receives it

function setReceives() {
  // mark them all as server, then adjust the client ones.
  // it would be nice to have some independent check
  req.forEach((_, k) => {receives.set(k, 'server')});
  not.forEach((_, k) => {receives.set(k, 'server')});
  receives.set('window/logMessage', 'client');
  receives.set('telemetry/event', 'client');
  receives.set('client/registerCapability', 'client');
  receives.set('client/unregisterCapability', 'client');
  receives.set('window/showMessage', 'client');
  receives.set('window/showMessageRequest', 'client');
  receives.set('workspace/workspaceFolders', 'client');
  receives.set('workspace/configuration', 'client');
  receives.set('workspace/applyEdit', 'client');
  receives.set('textDocument/publishDiagnostics', 'client');
  // a small check
  receives.forEach((_, k) => {
    if (!req.get(k) && !not.get(k)) throw new Error(`missing ${k}}`);
    if (req.get(k) && not.get(k)) throw new Error(`dup ${k}`);
  })
}

function goType(side: side, m: string, n: ts.Node): string {
  if (n === undefined) return '';
  if (ts.isTypeReferenceNode(n)) return n.typeName.getText();
  if (n.kind == ts.SyntaxKind.VoidKeyword) return '';
  if (n.kind == ts.SyntaxKind.AnyKeyword) return 'interface{}';
  if (ts.isArrayTypeNode(n)) return '[]' + goType(side, m, n.elementType);
  // special cases, before we get confused
  switch (m) {
    case 'textDocument/completion':
      return 'CompletionList';
    case 'textDocument/documentSymbol':
      return '[]DocumentSymbol';
    case 'textDocument/prepareRename':
      return 'Range';
    case 'textDocument/codeAction':
      return '[]CodeAction';
  }
  if (ts.isUnionTypeNode(n)) {
    let x: string[] = [];
    n.types.forEach(
        (v) => {v.kind != ts.SyntaxKind.NullKeyword &&
                x.push(goType(side, m, v))});
    if (x.length == 1) return x[0];

    prb(`===========${m} ${x}`)
    // Because we don't fully resolve types, we don't know that
    // Definition is Location | Location[]
    if (x[0] == 'Definition') return '[]Location';
    if (x[1] == '[]' + x[0] + 'Link') return x[1];
    throw new Error(`${m}, ${x} unexpected types`)
  }
  if (ts.isIntersectionTypeNode(n)) {
    // we expect only TypeReferences, and put out a struct with embedded types
    // This is not good, as it uses a struct where a type name ought to be.
    let x: string[] = [];
    n.types.forEach((v) => {
      // expect only TypeReferences
      if (!ts.isTypeReferenceNode(v)) {
        throw new Error(
            `expected only TypeReferences in Intersection ${getText(n)}`)
      }
      x.push(goType(side, m, v));
      x.push(';')
    })
    x.push('}')
    let ans = 'struct {'.concat(...x);
    // If ans does not have a type, create it
    if (side.ourTypes.get(ans) == undefined) {
      side.ourTypes.set(ans, 'Param' + getText(n).substring(0, 6))
    }
    // Return the type
    return side.ourTypes.get(ans)
  }
  return '?';
}

// walk the AST finding Requests and Notifications
function genStuff(node: ts.Node) {
  if (!ts.isNewExpression(node)) {
    ts.forEachChild(node, genStuff)
    return;
  }
  // process the right kind of new expression
  const wh = node.expression.getText();
  if (wh != 'RequestType' && wh != 'RequestType0' && wh != 'NotificationType' &&
      wh != 'NotificationType0')
    return;
  if (node.arguments === undefined || node.arguments.length != 1 ||
      !ts.isStringLiteral(node.arguments[0])) {
    throw new Error(`missing n.arguments ${loc(node)}`)
  }
  // RequestType<useful>=new RequestTYpe('foo')
  if (node.typeArguments === undefined) {
    node.typeArguments = lookUp(node);
  }
  // new RequestType<useful>
  let s = node.arguments[0].getText();
  // Request or Notification
  const v = wh[0] == 'R' ? req : not;
  s = s.substring(1, s.length - 1);    // remove quoting
  if (s == '$/cancelRequest') return;  // special case in output
  v.set(s, node);
}

// find the text of a node
function getText(node: ts.Node): string {
  let sf = node.getSourceFile();
  let start = node.getStart(sf)
  let end = node.getEnd()
  return sf.text.substring(start, end)
}

function lookUp(n: ts.NewExpression): ts.NodeArray<ts.TypeNode> {
  // parent should be VariableDeclaration. its children should be
  // Identifier('type') ???
  // TypeReference: [Identifier('RequestType1), ]
  // NewExpression (us)
  const p = n.parent;
  if (!ts.isVariableDeclaration(p)) throw new Error(`not variable decl`);
  const tr = p.type;
  if (!ts.isTypeReferenceNode(tr)) throw new Error(`not TypeReference`);
  return tr.typeArguments;
}

function dumpAST() {
  // dump the ast, for debugging
  for (const sourceFile of program.getSourceFiles()) {
    if (!sourceFile.isDeclarationFile) {
      // walk the tree to do stuff
      ts.forEachChild(sourceFile, describe);
    }
  }
}

// some tokens have the wrong default name
function strKind(n: ts.Node): string {
  const x = ts.SyntaxKind[n.kind];
  switch (x) {
    default:
      return x;
    case 'FirstAssignment':
      return 'EqualsToken';
    case 'FirstBinaryOperator':
      return 'LessThanToken';
    case 'FirstCompoundAssignment':
      return 'PlusEqualsToken';
    case 'FirstContextualKeyword':
      return 'AbstractKeyword';
    case 'FirstLiteralToken':
      return 'NumericLiteral';
    case 'FirstNode':
      return 'QualifiedName';
    case 'FirstTemplateToken':
      return 'NoSubstitutionTemplateLiteral';
    case 'LastTemplateToken':
      return 'TemplateTail';
    case 'FirstTypeNode':
      return 'TypePredicate';
  }
}

function describe(node: ts.Node) {
  if (node === undefined) {
    return
  }
  let indent = '';

  function f(n: ts.Node) {
    if (ts.isIdentifier(n)) {
      pra(`${indent} ${loc(n)} ${strKind(n)} ${n.text} \n`)
    } else if (ts.isPropertySignature(n) || ts.isEnumMember(n)) {
      pra(`${indent} ${loc(n)} ${strKind(n)} \n`)
    } else if (ts.isTypeLiteralNode(n)) {
      let m = n.members
      pra(`${indent} ${loc(n)} ${strKind(n)} ${m.length} \n`)
    } else {
      pra(`${indent} ${loc(n)} ${strKind(n)} \n`)
    };
    indent += '  '
    ts.forEachChild(n, f)
    indent = indent.slice(0, indent.length - 2)
  }
  f(node)
}

// string version of the location in the source file
function loc(node: ts.Node): string {
  const sf = node.getSourceFile();
  const start = node.getStart()
  const x = sf.getLineAndCharacterOfPosition(start)
  const full = node.getFullStart()
  const y = sf.getLineAndCharacterOfPosition(full)
  let fn = sf.fileName
  const n = fn.search(/-node./)
  fn = fn.substring(n + 6)
  return `${fn} ${x.line + 1}: ${x.character + 1} (${y.line + 1}: ${
      y.character + 1})`
}

// ad hoc argument parsing: [-d dir] [-o outputfile], and order matters
function main() {
  let args = process.argv.slice(2)  // effective command line
  if (args.length > 0) {
    let j = 0;
    if (args[j] == '-d') {
      dir = args[j + 1]
      j += 2
    }
    if (j != args.length) throw new Error(`incomprehensible args ${args}`)
  }
  let files: string[] = [];
  for (let i = 0; i < fnames.length; i++) {
    files.push(`${dir}${fnames[i]}`)
  }
  createOutputFiles()
  generate(
      files, {target: ts.ScriptTarget.ES5, module: ts.ModuleKind.CommonJS});
}

main()
