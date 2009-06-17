// Assembly constants

#define	g	R15
#define	m	R14

// offsets in m
#define	m_g0		0
#define	m_morepc	8
#define	m_morebuf	16
#define	m_morearg	40
#define	m_cret		48
#define	m_procid	56
#define	m_gsignal	64
#define	m_tls		72
#define	m_sched		104

// offsets in gobuf
#define	gobuf_sp	0
#define	gobuf_pc	8
#define	gobuf_g		16

// offsets in g
#define	g_stackguard	0
#define	g_stackbase	8
#define	g_defer		16
#define	g_sched		24
