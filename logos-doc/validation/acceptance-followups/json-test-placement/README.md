# Quoted-text test profile correction

The upstream append-ownership regression supplies the plain token
`{"v":"abc"}` to `QuotedTextPtr`. That is the JSONv2 method-input contract.
Classic `,string` text decoding requires an additional quoted string inside
the field value. The untagged placement therefore fails the classic profile
with `invalid use of ,string struct tag` before testing buffer ownership.

The preserved source-45 execution in `raw.log`, `inputs.json`, and
`result.json` records that setup failure. It is not a product RED and used
no production candidate. The fix moves the exact helper and test block from
the untagged scanner tests into the existing JSONv2-tagged test file.
No input, assertion, helper body, or production file changes in this commit.

The corrected stream test file is frozen as SHA-256
`08729986d528dd8a0c003893b0069639b5f3d7201b122a4105be162bfa088232`.
Its append control passes against source45; the independent permissive-Unicode
test remains the semantic RED for the following implementation integration.
Complete execution and mutation evidence is retained in the sibling
`json-text` record when that reviewed implementation is integrated.
