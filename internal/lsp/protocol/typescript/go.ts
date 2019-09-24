import * as fs from 'fs';
import * as ts from 'typescript';

// Need a general strategy for union types. This code tries (heuristically)
// to choose one, but sometimes defaults (heuristically) to interface{}
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
  embeds?: string[]
  fields?: Field[];
  extends?: string[]
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
const srcDir = '/vscode-languageserver-node'
let fnames = [
  `${srcDir}/protocol/src/protocol.ts`, `${srcDir}/types/src/main.ts`,
  `${srcDir}/jsonrpc/src/main.ts`
];
let gitHash = '36ac51f057215e6e2e0408384e07ecf564a938da';
let outFname = 'tsprotocol.go';
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

// struct names that don't need to go in the output
let dontEmit = new Map<string, boolean>();

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
    // Ignore top-level items that produce no output
    if (ts.isExpressionStatement(node) || ts.isFunctionDeclaration(node) ||
      ts.isImportDeclaration(node) || ts.isVariableStatement(node) ||
      ts.isExportDeclaration(node) || ts.isEmptyStatement(node) ||
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
      throw new Error(`unexpected ${strKind(node)} ${loc(node)}`)
    }
  }

  function doClassDeclaration(node: ts.ClassDeclaration) {
    let id: ts.Identifier = node.name;
    let props = new Array<ts.PropertyDeclaration>()
    let extend: ts.HeritageClause;
    let bad = false
    node.forEachChild((n: ts.Node) => {
      if (ts.isIdentifier(n)) {
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
        ts.isGetAccessor(n) || ts.isSetAccessor(n) ||
        ts.isTypeParameterDeclaration(n)) {
        bad = true;
        return
      }
      if (ts.isHeritageClause(n)) {
        return
      }
      if (n.kind == ts.SyntaxKind.AbstractKeyword) {
        bad = true;  // we think all of these are useless, but are unsure
        return;
      }
      throw new Error(`doClass ${loc(n)} ${kinds(n)}`)
    })
    if (bad) {
      // the class is not useful for Go.
      return
    }
    let fields: Field[] = [];
    for (const pr of props) {
      fields.push(fromPropDecl(pr))
    }
    let ans = {
      me: node,
      name: toGoName(getText(id)),
      extends: heritageStrs(extend),
      fields: fields
    };
    Structs.push(ans)
  }

  // called only from doClassDeclaration
  function fromPropDecl(node: ts.PropertyDeclaration): Field {
    let id: ts.Identifier = (ts.isIdentifier(node.name) && node.name);
    let opt = node.questionToken != undefined;
    let typ: ts.Node = node.type;
    const computed = computeType(typ);
    let goType = computed.goType
    let ans = {
      me: node,
      id: id,
      goName: toGoName(getText(id)),
      optional: opt,
      goType: goType,
      json: `\`json:"${id.text}${opt ? ',omitempty' : ''}"\``,
      substruct: computed.fields
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
    let id: ts.Identifier = node.name
    let extend: ts.HeritageClause
    let generid: ts.Identifier
    let properties = new Array<ts.PropertySignature>()
    let index: ts.IndexSignatureDeclaration  // generate some sort of map
    let bad = false;  // maybe we don't care about this one at all
    node.forEachChild((n: ts.Node) => {
      if (n.kind == ts.SyntaxKind.ExportKeyword || ts.isMethodSignature(n)) {
        // ignore
      } else if (ts.isIdentifier(n)) {
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
      } else if (n.kind == ts.SyntaxKind.CallSignature) {
        bad = true;
      } else {
        throw new Error(`${loc(n)} doInterface ${strKind(n)} `)
      }
    })
    if (bad) return;
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
      extends: heritageStrs(extend),
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

  // optional gen is the contents of <T>
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
    // These are not structs, so '*' would be wrong.
    switch (goType) {
      case 'CompletionItemKind':
      case 'TextDocumentSyncKind':
      case 'CodeActionKind':
      case 'FailureHandlingKind':  // string
      case 'InsertTextFormat':     // float64
      case 'DiagnosticSeverity':
        ans.optional = false
    }
    return ans
  }

  function doModuleDeclaration(node: ts.ModuleDeclaration) {
    // Export Identifier ModuleBlock
    let id: ts.Identifier = (ts.isIdentifier(node.name) && node.name);
    // Don't want FunctionDeclarations
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
    let id: ts.Identifier = (ts.isIdentifier(node.name) && node.name);
    let str: string
    let first: string
    node.forEachChild((n: ts.Node) => {
      if (ts.isStringLiteral(n)) {
        str = getText(n)
      } else if (n.kind == ts.SyntaxKind.NumericLiteral) {
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
    let id: ts.Identifier = node.name
    let mems = node.members
    let theType = 'string';
    for (const m of mems) {
      let name: string
      let value: string
      m.forEachChild((n: ts.Node) => {
        if (ts.isIdentifier(n)) {
          name = getText(n)
        } else if (ts.isStringLiteral(n)) {
          value = getText(n).replace(/'/g, '"')
        } else if (ts.isNumericLiteral(n)) {
          value = getText(n);
          theType = 'float64';
        } else {
          throw new Error(`in doEnumDecl ${strKind(n)} ${loc(n)}`)
        }
      })
      let ans = {
        typeName: getText(id),
        goType: theType,
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
    let id: ts.Identifier = node.name;
    let alias: ts.TypeNode = node.type;
    let ans = {
      me: node,
      id: id,
      goName: toGoName(getText(id)),
      goType: '?',  // filled in later in this function
      stuff: ''
    };
    if (ts.isUnionTypeNode(alias)) {
      ans.goType = weirdUnionType(alias)
      if (ans.goType == undefined) {
        // these are mostly redundant; maybe sort them out later
        return
      }
      if (ans.goType == 'interface{}') {
        // we didn't know what to do, so explain the choice
        ans.stuff = `// ` + getText(alias)
      }
      Types.push(ans)
      return
    }
    if (ts.isIntersectionTypeNode(alias)) {  // a Struct, not a Type
      let embeds: string[] = []
      alias.forEachChild((n: ts.Node) => {
        if (ts.isTypeReferenceNode(n)) {
          const s = toGoName(computeType(n).goType)
          embeds.push(s)
          // It's here just for embedding, and not used independently, maybe
          // PJW!
          // dontEmit.set(s, true); // PJW: do we need this?
        } else
          throw new Error(`expected TypeRef ${strKind(n)} ${loc(n)}`)
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
    if (ts.isTypeLiteralNode(alias)) {
      return;  // type A = {...}
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
    throw new Error(
      `in doTypeAlias ${loc(alias)} ${kinds(node)}: ${strKind(alias)}\n`)
  }

  // string, or number, or DocumentFilter
  function weirdUnionType(node: ts.UnionTypeNode): string {
    let bad = false;
    let aNumber = false;
    let aString = false;
    let tl: ts.TypeLiteralNode[] = []
    node.forEachChild((n: ts.Node) => {
      if (ts.isTypeLiteralNode(n)) {
        tl.push(n);
        return;
      }
      if (ts.isLiteralTypeNode(n)) {
        n.literal.kind == ts.SyntaxKind.NumericLiteral ? aNumber = true :
          aString = true;
        return;
      }
      if (n.kind == ts.SyntaxKind.NumberKeyword ||
        n.kind == ts.SyntaxKind.StringKeyword) {
        n.kind == ts.SyntaxKind.NumberKeyword ? aNumber = true : aString = true;
        return
      }
      bad = true
    })
    if (bad) return;  // none of these are useful (so far)
    if (aNumber) {
      if (aString) return 'interface{}';
      return 'float64';
    }
    if (aString) return 'string';
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

  // complex and filled with heuristics
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
        return { goType: toGoName(tn.text) };
      } else {
        throw new Error(
          `expected identifier got ${strKind(node.typeName)} at ${loc(tn)}`)
      }
    } else if (ts.isLiteralTypeNode(node)) {
      // string|float64 (are there other possibilities?)
      // as of 20190908: only see string
      const txt = getText(node);
      let typ = 'float64'
      if (txt.charAt(0) == '\'') {
        typ = 'string'
      }
      return { goType: typ, gostuff: getText(node) };
    } else if (ts.isTypeLiteralNode(node)) {
      // {[uri:string]: TextEdit[];} -> map[string][]TextEdit
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
        throw new Error(`${loc(n)} gotype ${strKind(n)}, not expected`)
      });
      if (indexCnt > 0) {
        if (indexCnt != 1 || x.length != 1)
          throw new Error(`undexpected Index ${loc(x[0].me)}`)
        // instead of {map...} just the map
        return ({ goType: x[0].goType, gostuff: x[0].gostuff })
      }
      return ({ goType: 'embedded!', fields: x })
    } else if (ts.isUnionTypeNode(node)) {
      // The major heuristics
      let x = new Array<{ goType: string, gostuff?: string, optiona?: boolean }>()
      node.forEachChild((n: ts.Node) => { x.push(computeType(n)) })
      if (x.length == 2 && x[1].goType == 'nil') {
        // Foo | null, or Foo | undefined
        return x[0]  // make it optional somehow? TODO
      }
      if (x[0].goType == 'bool') {  // take it, mostly
        if (x[1].goType == 'RenameOptions' ||
          x[1].goType == 'CodeActionOptions') {
          return ({ goType: 'interface{}', gostuff: getText(node) })
        }
        return ({ goType: 'bool', gostuff: getText(node) })
      }
      // these are special cases from looking at the source
      let gostuff = getText(node);
      if (x[0].goType == `"off"` || x[0].goType == 'string') {
        return ({ goType: 'string', gostuff: gostuff })
      }
      if (x[0].goType == 'TextDocumentSyncOptions') {
        // TextDocumentSyncOptions | TextDocumentSyncKind
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
      if (x[0].goType == 'RequestMessage' && x[1].goType == 'ResponseMessage') {
        return {
          goType: 'interface{}', gostuff: gostuff
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
      // DeleteFile) TODO(pjw) IT IS NOT! FIX THIS! ALSO:
      // (variousOptions & StaticFegistrationOptions)
      return {
        goType: 'TextDocumentEdit', gostuff: getText(node)
      }
    } else if (ts.isTupleTypeNode(node)) {
      // in string | [number, number]. TODO(pjw): check it really is
      return {
        goType: 'string', gostuff: getText(node)
      }
    } else if (ts.isFunctionTypeNode(node)) {
      // we don't want these members; mark them
      return {
        goType: 'bad', gostuff: getText(node)
      }
    }
    throw new Error(`computeType unknown ${strKind(node)} at ${loc(node)}`)
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
        throw new Error(`fromIndexSig ${strKind(n)} ${loc(n)}`)
    })
    let goType = computeType(at).goType
    let id: ts.Identifier
    parm.forEachChild((n: ts.Node) => {
      if (ts.isIdentifier(n)) {
        id = n
      } else if (n.kind != ts.SyntaxKind.StringKeyword) {
        throw new Error(`fromIndexSig expected string, ${strKind(n)} ${loc(n)}`)
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
    let res = 'Seen ' + strKind(n);
    function f(n: ts.Node): void { res += ' ' + strKind(n) };
    ts.forEachChild(n, f)
    return res
  }

  function strKind(n: ts.Node): string {
    const x = ts.SyntaxKind[n.kind];
    // some of these have two names
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
      seenAdd(kinds(n))
      if (ts.isIdentifier(n)) {
        pra(`${indent} ${loc(n)} ${strKind(n)} ${n.text}\n`)
      }
      else if (ts.isPropertySignature(n) || ts.isEnumMember(n)) {
        pra(`${indent} ${loc(n)} ${strKind(n)}\n`)
      }
      else if (ts.isTypeLiteralNode(n)) {
        let m = n.members
        pra(`${indent} ${loc(n)} ${strKind(n)} ${m.length}\n`)
      }
      else { pra(`${indent} ${loc(n)} ${strKind(n)}\n`) };
      indent += '  '
      ts.forEachChild(n, f)
      indent = indent.slice(0, indent.length - 2)
    }
    f(node)
  }
}

function getComments(node: ts.Node): string {
  const sf = node.getSourceFile();
  const start = node.getStart(sf, false)
  const starta = node.getStart(sf, true)
  const x = sf.text.substring(starta, start)
  return x
}

function loc(node: ts.Node): string {
  const sf = node.getSourceFile();
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

function emitTypes() {
  seenConstTypes.set('MessageQueue', true);  // skip
  for (const t of Types) {
    if (seenConstTypes.get(t.goName)) continue;
    if (t.goName == 'CodeActionKind') continue;  // consts better choice
    if (t.goType === undefined) continue;
    let stuff = (t.stuff == undefined) ? '' : t.stuff;
    prgo(`// ${t.goName} is a type\n`)
    prgo(`${getComments(t.me)}`)
    prgo(`type ${t.goName} = ${t.goType}${stuff}\n`)
    seenConstTypes.set(t.goName, true);
  }
}

let byName = new Map<string, Struct>();
function emitStructs() {
  dontEmit.set('Thenable', true);
  dontEmit.set('EmitterOptions', true);
  dontEmit.set('MessageReader', true);
  dontEmit.set('MessageWriter', true);
  dontEmit.set('CancellationToken', true);
  dontEmit.set('PipeTransport', true);
  dontEmit.set('SocketTransport', true);
  dontEmit.set('Item', true);
  dontEmit.set('Event', true);
  dontEmit.set('Logger', true);
  dontEmit.set('Disposable', true);
  dontEmit.set('PartialMessageInfo', true);
  dontEmit.set('MessageConnection', true);
  dontEmit.set('ResponsePromise', true);
  dontEmit.set('ResponseMessage', true);
  dontEmit.set('ErrorMessage', true);
  dontEmit.set('NotificationMessage', true);
  dontEmit.set('RequestHandlerElement', true);
  dontEmit.set('RequestMessage', true);
  dontEmit.set('NotificationHandlerElement', true);
  dontEmit.set('Message', true);  // duplicate of jsonrpc2:wire.go
  dontEmit.set('LSPLogMessage', true);
  dontEmit.set('InnerEM', true);
  dontEmit.set('ResponseErrorLiteral', true);
  dontEmit.set('TraceOptions', true);
  dontEmit.set('MessageType', true);  // want the enum
  // backwards compatibility, done in requests.ts:
  dontEmit.set('CancelParams', true);

  for (const str of Structs) {
    byName.set(str.name, str)
  }
  let seenName = new Map<string, boolean>()
  for (const str of Structs) {
    if (str.name == 'InitializeError') {
      // only want its consts, not the struct
      continue
    }
    if (seenName.get(str.name) || dontEmit.get(str.name)) {
      continue
    }
    let noopt = false;
    seenName.set(str.name, true)
    prgo(genComments(str.name, getComments(str.me)))
    prgo(`type ${str.name} struct {\n`)
    // if it has fields, generate them
    if (str.fields != undefined) {
      for (const f of str.fields) {
        prgo(strField(f, noopt))
      }
    }
    if (str.extends) {
      // ResourceOperation just repeats the Kind field
      for (const s of str.extends) {
        if (s != 'ResourceOperation')
          prgo(`\t${s}\n`)  // what this type extends.
      }
    } else if (str.embeds) {
      prb(`embeds: ${str.name}\n`);
      noopt = (str.name == 'ClientCapabilities');
      // embedded struct. the hard case is from intersection types,
      // where fields with the same name have to be combined into
      // a single struct
      let fields = new Map<string, Field[]>();
      for (const e of str.embeds) {
        const nm = byName.get(e);
        if (nm.embeds) throw new Error(`${nm.name} is an embedded embed`);
        // each of these fields might be a something that needs amalgamating
        for (const f of nm.fields) {
          let x = fields.get(f.goName);
          if (x === undefined) x = [];
          x.push(f);
          fields.set(f.goName, x);
        }
      }
      fields.forEach((val, key) => {
        if (val.length > 1) {
          // merge the fields with the same name
          prgo(strField(val[0], noopt, val));
        } else {
          prgo(strField(val[0], noopt));
        }
      });
    }
    prgo(`}\n`);
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
// flds is for merging
function strField(f: Field, noopt?: boolean, flds?: Field[]): string {
  let ans: string[] = [];
  let opt = (!noopt && f.optional) ? '*' : ''
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
  if (flds === undefined && f.substruct == undefined) {
    ans.push(`\t${f.goName} ${opt}${f.goType} ${f.json}${stuff}\n`)
  }
  else if (flds !== undefined) {
    // The logic that got us here is imprecise, so it is possible that
    // the fields are really all the same, and don't need to be
    // combined into a struct.
    let simple = true;
    for (const ff of flds) {
      if (ff.substruct !== undefined || byName.get(ff.goType) !== undefined) {
        simple = false
        break
      }
    }
    if (simple) {
      // should check that the ffs are really all the same
      return strField(flds[0], noopt)
    }
    ans.push(`\t${f.goName} ${opt}struct{\n`);
    for (const ff of flds) {
      if (ff.substruct !== undefined) {
        for (const x of ff.substruct) {
          ans.push(strField(x, noopt))
        }
      } else if (byName.get(ff.goType) !== undefined) {
        const st = byName.get(ff.goType);
        for (let i = 0; i < st.fields.length; i++) {
          ans.push(strField(st.fields[i], noopt))
        }
      } else {
        ans.push(strField(ff, noopt));
      }
    }
    ans.push(`\t} ${f.json}${stuff}\n`);
  }
  else {
    ans.push(`\t${f.goName} ${opt}struct {\n`)
    for (const x of f.substruct) {
      ans.push(strField(x, noopt))
    }
    ans.push(`\t} ${f.json}${stuff}\n`)
  }
  return (''.concat(...ans))
}

function emitConsts() {
  // need the consts too! Generate modifying prefixes and suffixes to ensure
  // consts are unique. (Go consts are package-level, but Typescript's are
  // not.) Use suffixes to minimize changes to gopls.
  let pref = new Map<string, string>([
    ['DiagnosticSeverity', 'Severity'], ['WatchKind', 'Watch']
  ])  // typeName->prefix
  let suff = new Map<string, string>([
    ['CompletionItemKind', 'Completion'], ['InsertTextFormat', 'TextFormat']
  ])
  for (const c of Consts) {
    if (seenConstTypes.get(c.typeName)) {
      continue
    }
    seenConstTypes.set(c.typeName, true);
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
    if (c.value === undefined) continue;      // didn't figure it out
    if (x.startsWith('undefined')) continue;  // what's going on here?
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
  let a = fs.readFileSync(`${dir}${srcDir}/.git/refs/heads/master`);
  prgo(`// Package protocol contains data types and code for LSP jsonrpcs\n`)
  prgo(`// generated automatically from vscode-languageserver-node\n`)
  prgo(`// commit: ${gitHash}\n`)
  prgo(`// last fetched ${lastDate}\n`)
  prgo('package protocol\n\n')
  prgo(`// Code generated (see typescript/README.md) DO NOT EDIT.\n`);
};

function git(): string {
  let a = fs.readFileSync(`${dir}${srcDir}/.git/HEAD`).toString();
  // ref: refs/heads/foo, or a hash like cc12d1a1c7df935012cdef5d085cdba04a7c8ebe
  if (a.charAt(a.length - 1) == '\n') {
    a = a.substring(0, a.length - 1);
  }
  if (a.length == 40) {
    return a // a hash
  }
  if (a.substring(0, 5) == 'ref: ') {
    const fname = `${dir}${srcDir}/.git/` + a.substring(5);
    let b = fs.readFileSync(fname).toString()
    if (b.length == 41) {
      return b.substring(0, 40);
    }
  }
  throw new Error("failed to find the git commit hash")
}

// ad hoc argument parsing: [-d dir] [-o outputfile], and order matters
function main() {
  if (gitHash != git()) {
    throw new Error(`git hash mismatch, wanted\n${gitHash} but source is at\n${git()}`)
  }
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
