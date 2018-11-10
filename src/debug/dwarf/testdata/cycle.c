typedef struct aaa *AAA;
typedef AAA BBB;
struct aaa { BBB val; };

AAA x(void) {
    return (AAA)0;
}
