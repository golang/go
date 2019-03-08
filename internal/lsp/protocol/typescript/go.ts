import * as fs from 'fs';
import * as ts from 'typescript';


interface Const {
  typeName: string  // repeated in each const
  goType: string
  me: ts.Node
  name: string   // constant's name
  value: string  // constant's value
}
let Consts: Const[] = [];
let seenConstTypes = new Map<string, boolean>();

interface Struct {
  me: ts.Node
  name: string
  embeds: string[]
  fields?: Field[]
}
let Structs: Struct[] = [];

interface Field {
  me: ts.Node
  id: ts.Identifier
  goName: string
  optional: boolean
  goType: string
  json: string
  gostuff?: string
  substruct?: Field[]  // embedded struct from TypeLiteral
}

interface Type {
  me: ts.Node
  goName: string
  goType: string
  stuff: string
}
let Types: Type[] = [];

// Used in printing the AST
let seenThings = new Map<string, number>();
function seenAdd(x: string) {
  seenThings[x] = (seenThings[x] === undefined ? 1 : seenThings[x] + 1)
}

let dir = process.env['HOME'];
let fnames = [
  `/vscode-languageserver-node/protocol/src/protocol.ts`,
  `/vscode-languageserver-node/types/src/main.ts`
];
let outFname = '/tmp/tsprotocol.go';
let fda: number, fdb: number, fde: number;  // file descriptors

function createOutputFiles() {
  fda = fs.openSync('/tmp/ts-a', 'w')  // dump of AST
  fdb = fs.openSync('/tmp/ts-b', 'w')  // unused, for debugging
  fde = fs.openSync(outFname, 'w')     // generated Go
}
function pra(s: string) {
  return (fs.writeSync(fda, s))
}
function prb(s: string) {
  return (fs.writeSync(fdb, s))
}
function prgo(s: string) {
  return (fs.writeSync(fde, s))
}

function generate(files: string[], options: ts.CompilerOptions): void {
  let program = ts.createProgram(files, options);
  program.getTypeChecker();  // used for side-effects

  // dump the ast, for debugging
  for (const sourceFile of program.getSourceFiles()) {
    if (!sourceFile.isDeclarationFile) {
      // walk the tree to do stuff
      ts.forEachChild(sourceFile, describe);
    }
  }
  pra('\n')
  for (const key of Object.keys(seenThings).sort()) {
    pra(`${key}: ${seenThings[key]}\n`)
  }

  // visit every sourceFile in the program, generating types
  for (const sourceFile of program.getSourceFiles()) {
    if (!sourceFile.isDeclarationFile) {
      ts.forEachChild(sourceFile, genTypes)
    }
  }
  return;

  function genTypes(node: ts.Node) {
    // Ignore top-level items with no output
    if (ts.isExpressionStatement(node) || ts.isFunctionDeclaration(node) ||
      ts.isImportDeclaration(node) || ts.isVariableStatement(node) ||
      ts.isExportDeclaration(node) ||
      node.kind == ts.SyntaxKind.EndOfFileToken) {
      return;
    }
    if (ts.isInterfaceDeclaration(node)) {
      doInterface(node)
      return;
    } else if (ts.isTypeAliasDeclaration(node)) {
      doTypeAlias(node)
    } else if (ts.isModuleDeclaration(node)) {
      doModuleDeclaration(node)
    } else if (ts.isEnumDeclaration(node)) {
      doEnumDecl(node)
    } else if (ts.isClassDeclaration(node)) {
      doClassDeclaration(node)
    } else {
      throw new Error(`unexpected ${ts.SyntaxKind[node.kind]} ${loc(node)}`)
    }
  }

  function doClassDeclaration(node: ts.ClassDeclaration) {
    let id: ts.Identifier
    let props = new Array<ts.PropertyDeclaration>()
    let extend: ts.HeritageClause;
    let bad = false
    node.forEachChild((n: ts.Node) => {
      if (ts.isIdentifier(n)) {
        id = n;
        return
      }
      if (ts.isPropertyDeclaration(n)) {
        props.push(n);
        return
      }
      if (n.kind == ts.SyntaxKind.ExportKeyword) {
        return
      }
      if (n.kind == ts.SyntaxKind.Constructor || ts.isMethodDeclaration(n) ||
        ts.isGetAccessor(n) || ts.isTypeParameterDeclaration(n)) {
        bad = true;
        return
      }
      if (ts.isHeritageClause(n)) {
        extend = n;
        return
      }
      throw new Error(`doClass ${loc(n)} ${kinds(n)}`)
    })
    if (bad) {
      // the class is not useful for Go.
      return
    }  // might we want the PropertyDecls? (don't think so)
    let fields: Field[] = [];
    for (const pr of props) {
      fields.push(fromPropDecl(pr))
    }
    let ans = {
      me: node,
      name: toGoName(getText(id)),
      embeds: heritageStrs(extend),
      fields: fields
    };
    Structs.push(ans)
  }

  function fromPropDecl(node: ts.PropertyDeclaration): Field {
    let id: ts.Identifier;
    let opt = false
    let typ: ts.Node
    node.forEachChild((n: ts.Node) => {
      if (ts.isIdentifier(n)) {
        id = n;
        return
      }
      if (n.kind == ts.SyntaxKind.QuestionToken) {
        opt = true;
        return
      }
      if (typ != undefined)
        throw new Error(`fromPropDecl too long ${loc(node)}`)
      typ = n
    })
    let goType = computeType(typ).goType
    let ans = {
      me: node,
      id: id,
      goName: toGoName(getText(id)),
      optional: opt,
      goType: goType,
      json: `\`json:"${id.text}${opt ? ',omitempty' : ''}"\``
    };
    return ans
  }

  function doInterface(node: ts.InterfaceDeclaration) {
    // name: Identifier;
    // typeParameters?: NodeArray<TypeParameterDeclaration>;
    // heritageClauses?: NodeArray<HeritageClause>;
    // members: NodeArray<TypeElement>;

    // find the Identifier from children
    // process the PropertySignature children
    // the members might have generic info, but so do the children
    let id: ts.Identifier;
    let extend: ts.HeritageClause
    let generid: ts.Identifier
    let properties = new Array<ts.PropertySignature>()
    let index: ts.IndexSignatureDeclaration  // generate some sort of map
    node.forEachChild((n: ts.Node) => {
      if (n.kind == ts.SyntaxKind.ExportKeyword || ts.isMethodSignature(n)) {
        // ignore
      } else if (ts.isIdentifier(n)) {
        id = n;
      } else if (ts.isHeritageClause(n)) {
        extend = n;
      } else if (ts.isTypeParameterDeclaration(n)) {
        // Act as if this is <T = any>
        generid = n.name;
      } else if (ts.isPropertySignature(n)) {
        properties.push(n);
      } else if (ts.isIndexSignatureDeclaration(n)) {
        if (index !== undefined) {
          throw new Error(`${loc(n)} multiple index expressions`)
        }
        index = n
      } else {
        throw new Error(`${loc(n)} doInterface ${ts.SyntaxKind[n.kind]} `)
      }
    })
    let fields: Field[] = [];
    for (const p of properties) {
      fields.push(genProp(p, generid))
    }
    if (index != undefined) {
      fields.push(fromIndexSignature(index))
    }
    const ans = {
      me: node,
      name: toGoName(getText(id)),
      embeds: heritageStrs(extend),
      fields: fields
    };

    Structs.push(ans)
  }

  function heritageStrs(node: ts.HeritageClause): string[] {
    // ExpressionWithTypeArguments+, and each is an Identifier
    let ans: string[] = [];
    if (node == undefined) {
      return ans
    }
    let x: ts.ExpressionWithTypeArguments[] = []
    node.forEachChild((n: ts.Node) => {
      if (ts.isExpressionWithTypeArguments(n)) x.push(n)
    })
    for (const p of x) {
      p.forEachChild((n: ts.Node) => {
        if (ts.isIdentifier(n)) {
          ans.push(toGoName(getText(n)));
          return;
        }
        if (ts.isTypeReferenceNode(n)) {
          // don't want these, ignore them
          return;
        }
        throw new Error(`expected Identifier ${loc(n)} ${kinds(p)} `)
      })
    }
    return ans
  }

  function genProp(node: ts.PropertySignature, gen: ts.Identifier): Field {
    let id: ts.Identifier
    let thing: ts.Node
    let opt = false
    node.forEachChild((n: ts.Node) => {
      if (ts.isIdentifier(n)) {
        id = n
      } else if (n.kind == ts.SyntaxKind.QuestionToken) {
        opt = true
      } else if (n.kind == ts.SyntaxKind.ReadonlyKeyword) {
        return
      } else {
        if (thing !== undefined) {
          throw new Error(`${loc(n)} weird`)
        }
        thing = n
      }
    })
    let goName = toGoName(id.text)
    let { goType, gostuff, optional, fields } = computeType(thing)
    // Generics
    if (gen && gen.text == goType) goType = 'interface{}';
    opt = opt || optional;
    let ans = {
      me: node,
      id: id,
      goName: goName,
      optional: opt,
      goType: goType,
      gostuff: gostuff,
      substruct: fields,
      json: `\`json:"${id.text}${opt ? ',omitempty' : ''}"\``
    };
    // Rather than checking that goName is a const type, just do
    switch (goType) {
      case 'CompletionItemKind':
      case 'TextDocumentSyncKind':
      case 'CodeActionKind':
      case 'InsertTextFormat':  // float64
      case 'DiagnosticSeverity':
        ans.optional = false
    }
    return ans
  }

  function doModuleDeclaration(node: ts.ModuleDeclaration) {
    // Export Identifier ModuleBlock
    let id: ts.Identifier;
    let mb: ts.ModuleBlock;
    node.forEachChild((n: ts.Node) => {
      if ((ts.isIdentifier(n) && (id = n)) ||
        (ts.isModuleBlock(n) && mb === undefined && (mb = n)) ||
        (n.kind == ts.SyntaxKind.ExportKeyword)) {
        return;
      }
      throw new Error(`doModuleDecl ${loc(n)} ${ts.SyntaxKind[n.kind]}`)
    })
    // Don't want FunctionDeclarations
    // mb has VariableStatement and useless TypeAliasDeclaration
    // some of the VariableStatement are consts, and want their comments
    // and each VariableStatement is Export, VariableDeclarationList
    // and each VariableDeclarationList is a single VariableDeclaration
    let v: ts.VariableDeclaration[] = [];
    function f(n: ts.Node) {
      if (ts.isVariableDeclaration(n)) {
        v.push(n);
        return
      }
      if (ts.isFunctionDeclaration(n)) {
        return
      }
      n.forEachChild(f)
    }
    f(node)
    for (const vx of v) {
      if (hasNewExpression(vx)) {
        return
      }
      buildConst(getText(id), vx)
    }
  }

  function buildConst(tname: string, node: ts.VariableDeclaration): Const {
    // node is Identifier, optional-goo, (FirstLiteralToken|StringLiteral)
    let id: ts.Identifier
    let str: string
    let first: string
    node.forEachChild((n: ts.Node) => {
      if (ts.isIdentifier(n)) {
        id = n
      } else if (ts.isStringLiteral(n)) {
        str = getText(n)
      } else if (n.kind == ts.SyntaxKind.FirstLiteralToken) {
        first = getText(n)
      }
    })
    if (str == undefined && first == undefined) {
      return
    }  // various
    const ty = (str != undefined) ? 'string' : 'float64'
    const val = (str != undefined) ? str.replace(/'/g, '"') : first
    const name = toGoName(getText(id))
    const c = {
      typeName: tname,
      goType: ty,
      me: node.parent.parent,
      name: name,
      value: val
    };
    Consts.push(c)
    return c
  }

  // is node an ancestor of a NewExpression
  function hasNewExpression(n: ts.Node): boolean {
    let ans = false;
    n.forEachChild((n: ts.Node) => {
      if (ts.isNewExpression(n)) ans = true;
    })
    return ans
  }

  function doEnumDecl(node: ts.EnumDeclaration) {
    // Generates Consts. Identifier EnumMember+
    // EnumMember: Identifier StringLiteral
    let id: ts.Identifier
    let mems: ts.EnumMember[] = []
    node.forEachChild((n: ts.Node) => {
      if (ts.isIdentifier(n)) {
        id = n  // check for uniqueness?
      } else if (ts.isEnumMember(n)) {
        mems.push(n)
      } else if (n.kind != ts.SyntaxKind.ExportKeyword) {
        throw new Error(`doEnumDecl ${ts.SyntaxKind[n.kind]} ${loc(n)}`)
      }
    })
    for (const m of mems) {
      let name: string
      let value: string
      m.forEachChild((n: ts.Node) => {
        if (ts.isIdentifier(n)) {
          name = getText(n)
        } else if (ts.isStringLiteral(n)) {
          value = getText(n).replace(/'/g, '"')
        } else {
          throw new Error(`in doEnumDecl ${ts.SyntaxKind[n.kind]} ${loc(n)}`)
        }
      })
      let ans = {
        typeName: getText(id),
        goType: 'string',
        me: m,
        name: name,
        value: value
      };
      Consts.push(ans)
    }
  }

  // top-level TypeAlias
  function doTypeAlias(node: ts.TypeAliasDeclaration) {
    // these are all Export Identifier alias
    let id: ts.Identifier;
    let alias: ts.Node;
    let genid: ts.TypeParameterDeclaration  // <T>, but we don't care
    node.forEachChild((n: ts.Node) => {
      if ((ts.isIdentifier(n) && (id = n)) ||
        (n.kind == ts.SyntaxKind.ExportKeyword) ||
        ts.isTypeParameterDeclaration(n) && (genid = n) ||
        (alias === undefined && (alias = n))) {
        return
      }
      throw new Error(`doTypeAlias ${loc(n)} ${ts.SyntaxKind[n.kind]}`)
    })
    let ans = {
      me: node,
      id: id,
      goName: toGoName(getText(id)),
      goType: '?',
      stuff: ''
    };
    if (id.text.indexOf('--') != -1) {
      return
    }  // don't care
    if (ts.isUnionTypeNode(alias)) {
      ans.goType = weirdUnionType(alias)
      if (ans.goType == undefined) {  // these are redundant
        return
      }
      Types.push(ans)
      return
    }
    if (ts.isIntersectionTypeNode(alias)) {  // a Struct, not a Type
      let embeds: string[] = []
      alias.forEachChild((n: ts.Node) => {
        if (ts.isTypeReferenceNode(n)) {
          embeds.push(toGoName(computeType(n).goType))
        } else
          throw new Error(`expected TypeRef ${ts.SyntaxKind[n.kind]} ${loc(n)}`)
      })
      let ans = { me: node, name: toGoName(getText(id)), embeds: embeds };
      Structs.push(ans)
      return
    }
    if (ts.isArrayTypeNode(alias)) {  // []DocumentFilter
      ans.goType = '[]DocumentFilter';
      Types.push(ans)
      return
    }
    if (ts.isLiteralTypeNode(alias)) {
      return  // type A = 1, so nope
    }
    if (ts.isTypeReferenceNode(alias)) {
      ans.goType = computeType(alias).goType
      if (ans.goType.match(/und/) != null) throw new Error('396')
      Types.push(ans)  // type A B
      return
    }
    if (alias.kind == ts.SyntaxKind.StringKeyword) {  // type A string
      ans.goType = 'string';
      Types.push(ans);
      return
    }
    throw new Error(`in doTypeAlias ${loc(node)} ${kinds(node)} ${
      ts.SyntaxKind[alias.kind]}\n`)
  }

  // extract the one useful but weird case ()
  function weirdUnionType(node: ts.UnionTypeNode): string {
    let bad = false
    let tl: ts.TypeLiteralNode[] = []
    node.forEachChild((n: ts.Node) => {
      if (ts.isTypeLiteralNode(n)) {
        tl.push(n)
      } else
        bad = true
    })
    if (bad) return  // none of these are useful (so far)
    let x = computeType(tl[0])
    x.fields[0].json = x.fields[0].json.replace(/"`/, ',omitempty"`')
    let out: string[] = [];
    for (const f of x.fields) {
      out.push(strField(f))
    }
    out.push('}\n')
    let ans = 'struct {\n'.concat(...out);
    return ans
  }

  function computeType(node: ts.Node): { goType: string, gostuff?: string, optional?: boolean, fields?: Field[] } {
    switch (node.kind) {
      case ts.SyntaxKind.AnyKeyword:
      case ts.SyntaxKind.ObjectKeyword:
        return { goType: 'interface{}' };
      case ts.SyntaxKind.BooleanKeyword:
        return { goType: 'bool' };
      case ts.SyntaxKind.NumberKeyword:
        return { goType: 'float64' };
      case ts.SyntaxKind.StringKeyword:
        return { goType: 'string' };
      case ts.SyntaxKind.NullKeyword:
      case ts.SyntaxKind.UndefinedKeyword:
        return { goType: 'nil' };
    }
    if (ts.isArrayTypeNode(node)) {
      let { goType, gostuff, optional } = computeType(node.elementType)
      return ({ goType: '[]' + goType, gostuff: gostuff, optional: optional })
    } else if (ts.isTypeReferenceNode(node)) {
      // typeArguments?: NodeArray<TypeNode>;typeName: EntityName;
      // typeArguments won't show up in the generated Go
      // EntityName: Identifier|QualifiedName
      let tn: ts.EntityName = node.typeName;
      if (ts.isQualifiedName(tn)) {
        throw new Error(`qualified name at ${loc(node)}`);
      } else if (ts.isIdentifier(tn)) {
        return { goType: tn.text };
      } else {
        throw new Error(`expected identifier got ${
          ts.SyntaxKind[node.typeName.kind]} at ${loc(tn)}`)
      }
    } else if (ts.isLiteralTypeNode(node)) {
      // string|float64 (are there other possibilities?)
      const txt = getText(node);
      let typ = 'float64'
      if (txt.charAt(0) == '\'') {
        typ = 'string'
      }
      return { goType: typ, gostuff: getText(node) };
    } else if (ts.isTypeLiteralNode(node)) {
      let x: Field[] = [];
      let indexCnt = 0
      node.forEachChild((n: ts.Node) => {
        if (ts.isPropertySignature(n)) {
          x.push(genProp(n, undefined))
          return
        } else if (ts.isIndexSignatureDeclaration(n)) {
          indexCnt++
          x.push(fromIndexSignature(n))
          return
        }
        throw new Error(
          `${loc(n)} gotype ${ts.SyntaxKind[n.kind]}, not expected`)
      });
      if (indexCnt > 0) {
        if (indexCnt != 1 || x.length != 1)
          throw new Error(`undexpected Index ${loc(x[0].me)}`)
        // instead of {map...} just the map
        return ({ goType: x[0].goType, gostuff: x[0].gostuff })
      }
      return ({ goType: 'embedded!', fields: x })
    } else if (ts.isUnionTypeNode(node)) {
      let x = new Array<{ goType: string, gostuff?: string, optiona?: boolean }>()
      node.forEachChild((n: ts.Node) => { x.push(computeType(n)) })
      if (x.length == 2 && x[1].goType == 'nil') {
        return x[0]  // make it optional somehow? TODO
      }
      if (x[0].goType == 'bool') {  // take it
        return ({ goType: 'bool', gostuff: getText(node) })
      }
      // these are special cases from looking at the source
      let gostuff = getText(node);
      if (x[0].goType == `"off"` || x[0].goType == 'string') {
        return ({ goType: 'string', gostuff: gostuff })
      }
      if (x[0].goType == 'TextDocumentSyncOptions') {
        return ({ goType: 'interface{}', gostuff: gostuff })
      }
      if (x[0].goType == 'float64' && x[1].goType == 'string') {
        return {
          goType: 'interface{}', gostuff: gostuff
        }
      }
      if (x[0].goType == 'MarkupContent' && x[1].goType == 'MarkedString') {
        return {
          goType: 'MarkupContent', gostuff: gostuff
        }
      }
      // Fail loudly
      console.log(`UnionType ${loc(node)}`)
      for (const v of x) {
        console.log(`${v.goType}`)
      }
      throw new Error('in UnionType, weird')
    } else if (ts.isParenthesizedTypeNode(node)) {
      // check that this is (TextDocumentEdit | CreateFile | RenameFile |
      // DeleteFile)
      return {
        goType: 'TextDocumentEdit', gostuff: getText(node)
      }
    } else if (ts.isTupleTypeNode(node)) {
      // string | [number, number]
      return {
        goType: 'string', gostuff: getText(node)
      }
    }
    throw new Error(`unknown ${ts.SyntaxKind[node.kind]} at ${loc(node)}`)
  }

  function fromIndexSignature(node: ts.IndexSignatureDeclaration): Field {
    let parm: ts.ParameterDeclaration
    let at: ts.Node
    node.forEachChild((n: ts.Node) => {
      if (ts.isParameter(n)) {
        parm = n
      } else if (
        ts.isArrayTypeNode(n) || n.kind == ts.SyntaxKind.AnyKeyword ||
        ts.isUnionTypeNode(n)) {
        at = n
      } else
        throw new Error(`fromIndexSig ${ts.SyntaxKind[n.kind]} ${loc(n)}`)
    })
    let goType = computeType(at).goType
    let id: ts.Identifier
    parm.forEachChild((n: ts.Node) => {
      if (ts.isIdentifier(n)) {
        id = n
      } else if (n.kind != ts.SyntaxKind.StringKeyword) {
        throw new Error(
          `fromIndexSig expected string, ${ts.SyntaxKind[n.kind]} ${loc(n)}`)
      }
    })
    goType = `map[string]${goType}`
    return {
      me: node, goName: toGoName(id.text), id: null, goType: goType,
      optional: false, json: `\`json:"${id.text}"\``,
      gostuff: `${getText(node)}`
    }
  }

  function toGoName(s: string): string {
    let ans = s
    if (s.charAt(0) == '_') {
      ans = 'Inner' + s.substring(1)
    }
    else { ans = s.substring(0, 1).toUpperCase() + s.substring(1) };
    ans = ans.replace(/Uri$/, 'URI')
    ans = ans.replace(/Id$/, 'ID')
    return ans
  }


  // find the text of a node
  function getText(node: ts.Node): string {
    let sf = node.getSourceFile();
    let start = node.getStart(sf)
    let end = node.getEnd()
    return sf.text.substring(start, end)
  }
  // return a string of the kinds of the immediate descendants
  function kinds(n: ts.Node): string {
    let res = 'Seen ' + ts.SyntaxKind[n.kind];
    function f(n: ts.Node): void { res += ' ' + ts.SyntaxKind[n.kind] };
    ts.forEachChild(n, f)
    return res
  }

  function describe(node: ts.Node) {
    if (node === undefined) {
      return
    }
    let indent = '';

    function f(n: ts.Node) {
      seenAdd(kinds(n))
      if (ts.isIdentifier(n)) {
        pra(`${indent} ${loc(n)} ${ts.SyntaxKind[n.kind]} ${n.text}\n`)
      }
      else if (ts.isPropertySignature(n) || ts.isEnumMember(n)) {
        pra(`${indent} ${loc(n)} ${ts.SyntaxKind[n.kind]}\n`)
      }
      else if (ts.isTypeLiteralNode(n)) {
        let m = n.members
        pra(`${indent} ${loc(n)} ${ts.SyntaxKind[n.kind]} ${m.length}\n`)
      }
      else { pra(`${indent} ${loc(n)} ${ts.SyntaxKind[n.kind]}\n`) };
      indent += '  '
      ts.forEachChild(n, f)
      indent = indent.slice(0, indent.length - 2)
    }
    f(node)
  }


  function loc(node: ts.Node): string {
    const sf = node.getSourceFile()
    const start = node.getStart()
    const x = sf.getLineAndCharacterOfPosition(start)
    const full = node.getFullStart()
    const y = sf.getLineAndCharacterOfPosition(full)
    let fn = sf.fileName
    const n = fn.search(/-node./)
    fn = fn.substring(n + 6)
    return `${fn} ${x.line + 1}:${x.character + 1} (${y.line + 1}:${
      y.character + 1})`
  }
}

function getComments(node: ts.Node): string {
  const sf = node.getSourceFile();
  const start = node.getStart(sf, false)
  const starta = node.getStart(sf, true)
  const x = sf.text.substring(starta, start)
  return x
}

function emitTypes() {
  for (const t of Types) {
    if (t.goName == 'CodeActionKind') continue;  // consts better choice
    let stuff = (t.stuff == undefined) ? '' : t.stuff;
    prgo(`// ${t.goName} is a type\n`)
    prgo(`${getComments(t.me)}`)
    prgo(`type ${t.goName} ${t.goType}${stuff}\n`)
  }
}

function emitStructs() {
  let seenName = new Map<string, boolean>()
  for (const str of Structs) {
    if (str.name == 'InitializeError') {
      // only want the consts
      continue
    }
    if (seenName[str.name]) {
      continue
    }
    seenName[str.name] = true
    prgo(genComments(str.name, getComments(str.me)))
    /* prgo(`// ${str.name} is:\n`)
    prgo(getComments(str.me))*/
    prgo(`type ${str.name} struct {\n`)
    for (const s of str.embeds) {
      prgo(`\t${s}\n`)
    }
    if (str.fields != undefined) {
      for (const f of str.fields) {
        prgo(strField(f))
      }
    }
    prgo(`}\n`)
  }
}

function genComments(name: string, maybe: string): string {
  if (maybe == '') return `\n\t// ${name} is\n`;
  if (maybe.indexOf('/**') == 0) {
    return maybe.replace('/**', `\n/*${name} defined:`)
  }
  throw new Error(`weird comment ${maybe.indexOf('/**')}`)
}

// Turn a Field into an output string
function strField(f: Field): string {
  let ans: string[] = [];
  let opt = f.optional ? '*' : ''
  switch (f.goType.charAt(0)) {
    case 's':  // string
    case 'b':  // bool
    case 'f':  // float64
    case 'i':  // interface{}
    case '[':  // []foo
      opt = ''
  }
  let stuff = (f.gostuff == undefined) ? '' : ` // ${f.gostuff}`
  ans.push(genComments(f.goName, getComments(f.me)))
  if (f.substruct == undefined) {
    ans.push(`\t${f.goName} ${opt}${f.goType} ${f.json}${stuff}\n`)
  }
  else {
    ans.push(`\t${f.goName} ${opt}struct {\n`)
    for (const x of f.substruct) {
      ans.push(strField(x))
    }
    ans.push(`\t} ${f.json}${stuff}\n`)
  }
  return (''.concat(...ans))
}

function emitConsts() {
  // Generate modifying prefixes and suffixes to ensure consts are
  // unique. (Go consts are package-level, but Typescript's are not.)
  // Use suffixes to minimize changes to gopls.
  let pref = new Map<string, string>(
    [['DiagnosticSeverity', 'Severity']])  // typeName->prefix
  let suff = new Map<string, string>([
    ['CompletionItemKind', 'Completion'], ['InsertTextFormat', 'TextFormat']
  ])
  for (const c of Consts) {
    if (seenConstTypes[c.typeName]) {
      continue
    }
    seenConstTypes[c.typeName] = true
    if (pref.get(c.typeName) == undefined) {
      pref.set(c.typeName, '')  // initialize to empty value
    }
    if (suff.get(c.typeName) == undefined) {
      suff.set(c.typeName, '')
    }
    prgo(`// ${c.typeName} defines constants\n`)
    prgo(`type ${c.typeName} ${c.goType}\n`)
  }
  prgo('const (\n')
  let seenConsts = new Map<string, boolean>()  // to avoid duplicates
  for (const c of Consts) {
    const x = `${pref.get(c.typeName)}${c.name}${suff.get(c.typeName)}`
    if (seenConsts.get(x)) {
      continue
    }
    seenConsts.set(x, true)
    prgo(genComments(x, getComments(c.me)))
    prgo(`\t${x} ${c.typeName} = ${c.value}\n`)
  }
  prgo(')\n')
}

function emitHeader(files: string[]) {
  let lastMod = 0
  let lastDate: Date
  for (const f of files) {
    const st = fs.statSync(f)
    if (st.mtimeMs > lastMod) {
      lastMod = st.mtimeMs
      lastDate = st.mtime
    }
  }
  prgo(`// Package protocol contains data types for LSP jsonrpcs\n`)
  prgo(`// generated automatically from vscode-languageserver-node
  //  version of ${lastDate}\n`)
  prgo('package protocol\n\n')
};

// ad hoc argument parsing: [-d dir] [-o outputfile], and order matters
function main() {
  let args = process.argv.slice(2)  // effective command line
  if (args.length > 0) {
    let j = 0;
    if (args[j] == '-d') {
      dir = args[j + 1]
      j += 2
    }
    if (args[j] == '-o') {
      outFname = args[j + 1]
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
    files, { target: ts.ScriptTarget.ES5, module: ts.ModuleKind.CommonJS });
  emitHeader(files)
  emitStructs()
  emitConsts()
  emitTypes()
}

main()
