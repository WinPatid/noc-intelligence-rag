/**
 * NOC Intelligence Portfolio — Interactive Client Controller
 * Provides modal lightbox, interactive query inspector, and smooth navigation.
 * Zero external dependencies.
 */

document.addEventListener('DOMContentLoaded', () => {
  initLightbox();
  initSimulator();
  initCopyButtons();
  initMobileNav();
  initScrollSpy();
});

// ─── 1. Screenshot Lightbox Modal ───────────────────────────────────────────
function initLightbox() {
  const modal = document.getElementById('screenshotModal');
  const modalImg = document.getElementById('modalImage');
  const modalTitle = document.getElementById('modalTitle');
  const modalCaption = document.getElementById('modalCaption');
  const closeBtn = document.getElementById('modalClose');

  if (!modal) return;

  const cards = document.querySelectorAll('.gallery-card');
  cards.forEach(card => {
    card.addEventListener('click', () => {
      const img = card.querySelector('img');
      const title = card.querySelector('.gallery-title')?.innerText || 'Screenshot View';
      const desc = card.querySelector('.gallery-desc')?.innerText || '';

      if (img) {
        modalImg.src = img.src;
        modalImg.alt = img.alt;
        modalTitle.innerText = title;
        modalCaption.innerText = desc;
        modal.classList.add('active');
        document.body.style.overflow = 'hidden';
      }
    });
  });

  const closeModal = () => {
    modal.classList.remove('active');
    document.body.style.overflow = '';
  };

  if (closeBtn) closeBtn.addEventListener('click', closeModal);
  modal.addEventListener('click', (e) => {
    if (e.target === modal) closeModal();
  });

  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && modal.classList.contains('active')) {
      closeModal();
    }
  });
}

// ─── 2. Interactive Query Inspector Simulator ───────────────────────────────
const SIM_SCENARIOS = {
  ticket: {
    query: "Ticket INC-2025-00001 สถานะเป็นอะไร",
    route: "SQL_LOOKUP",
    intent: "TICKET_STATUS",
    target: "SQLite: trouble_tickets (SELECT * FROM trouble_tickets WHERE ticket_id = ?)",
    latency: "6.55 ms (E2E)",
    status: "VERIFIED",
    badgeClass: "score-success",
    evidence: `[SOURCE: ticket:INC-2025-00001] [DATABASE FACT (Exact SQL Record)]
Ticket ID: INC-2025-00001
Customer: บริษัท ซีพี ออลล์ จำกัด (มหาชน) (CP ALL PUBLIC COMPANY LIMITED)
Circuit: CKT-880008 - Terminal East
Status: RESOLVED
Severity: P1-Critical
Symptom: Optical port link down alarm on interface GigabitEthernet0/0/1`,
    maskedOutput: `🛡️ VERIFIED EVIDENCE  ⏱️ 6.55 ms

จากข้อมูลในระบบ [SOURCE: ticket:[TICKET-***001]] ตั๋วปัญหา [TICKET-***001] ของลูกค้า [CUSTOMER-XI] ([CUSTOMER-XI]) มีสถานะปัจจุบันเป็น RESOLVED ระดับความรุนแรง P1-Critical บนวงจร [CIRCUIT-***008] โดยมีอาการที่ได้รับแจ้งคือ: Optical port link down alarm on interface GigabitEthernet0/0/1 / พอร์ตสัญญาณดาวน์

📚 Verified Sources:
[1] [TICKET-***001] (trouble_ticket) ✅ Verified — Ticket [TICKET-***001] (RESOLVED)`
  },

  workorder: {
    query: "Show resolution for work order WO-700165",
    route: "SQL_LOOKUP",
    intent: "WORK_ORDER_DETAIL",
    target: "SQLite: work_orders (SELECT * FROM work_orders WHERE work_order_id = ?)",
    latency: "2.86 ms (E2E)",
    status: "VERIFIED",
    badgeClass: "score-success",
    evidence: `[SOURCE: work_order:WO-700165] [DATABASE FACT (Exact SQL Record)]
Work Order ID: WO-700165
Status: COMPLETED
Root Cause Category: CONFIGURATION_ERROR (การกำหนดค่าคอนฟิกูเรชันผิดพลาด)
Specific Cause: Mismatched MTU setting on sub-interface and route-map filter dropped customer Jumbo frames
Action Taken: Corrected MTU configuration to 9216 on PE sub-interface; verified BGP route-policy filters
Assigned Team: Tier-2 IP Core NOC`,
    maskedOutput: `🛡️ VERIFIED EVIDENCE  ⏱️ 2.86 ms

According to verified system records [SOURCE: work_order:[WO-***165]], work order [WO-***165] is in COMPLETED status with root cause category CONFIGURATION_ERROR (การกำหนดค่าคอนฟิกูเรชันผิดพลาด). Specific cause: Mismatched MTU setting on sub-interface and route-map filter dropped customer Jumbo frames. Action taken: Corrected MTU configuration to 9216 on PE sub-interface; verified BGP route-policy filters. Assigned team: Tier-2 IP Core NOC.

📚 Verified Sources:
[1] [WO-***165] (work_order) ✅ Verified — Work Order [WO-***165] (COMPLETED)`
  },

  bridge: {
    query: "ตั๋ว INC-2025-00042 เชื่อมโยงกับใบงานช่างใบไหนบ้าง",
    route: "SQL_LOOKUP",
    intent: "BRIDGE_TICKET_TO_WO",
    target: "SQLite: ticket_work_order_links JOIN work_orders (Traversed Bridge 1:M)",
    latency: "5.49 ms (E2E)",
    status: "VERIFIED",
    badgeClass: "score-success",
    evidence: `[SOURCE: ticket:INC-2025-00042] [DATABASE FACT (Exact SQL Record)]
Ticket ID: INC-2025-00042 | Status: CLOSED
Total Linked Work Orders: 1

[SOURCE: work_order:WO-700042] [LINKED WORK ORDER (Traversed Bridge)]
Work Order ID: WO-700042 (Relationship: PRIMARY_RESOLUTION)
Status: CLOSED | Root Cause: FIBER_CUT (สายไฟเบอร์ออปติกภายนอกขาดชำรุด)
Action Taken: OSP splicing crew dispatched. Performed fusion splicing on 48 cores, OTDR tested within normal optical limits`,
    maskedOutput: `🛡️ VERIFIED EVIDENCE  ⏱️ 5.49 ms

จากข้อมูลในระบบ [SOURCE: ticket:[TICKET-***042]] ตั๋วปัญหา [TICKET-***042] (สถานะ: CLOSED) เชื่อมโยงกับใบงานช่าง 1 รายการ:

- ใบงาน [WO-***042] [SOURCE: work_order:[WO-***042]] (สถานะ: CLOSED): สาเหตุหลัก FIBER_CUT (สายไฟเบอร์ออปติกภายนอกขาดชำรุด) การแก้ไข: OSP splicing crew dispatched. Performed fusion splicing on 48 cores, OTDR tested within normal optical limits มอบหมายทีม Transmission

📚 Verified Sources:
[1] [TICKET-***042] (trouble_ticket) ✅ Verified
[2] [WO-***042] (work_order) ✅ Verified`
  },

  greeting_en: {
    query: "Hello! How can you help me?",
    route: "CLARIFICATION",
    intent: "CLARIFICATION_GREETING",
    target: "Direct Conversational Capabilities Guide (Zero DB execution)",
    latency: "0.48 ms (E2E)",
    status: "VERIFIED",
    badgeClass: "score-success",
    evidence: `[ROUTER PRECEDENCE RULE 8b: RE_GREETING MATCH]
Query identified as conversational greeting or capability inquiry.
Target: Guided clarification prompt introduce telecommunications domain capabilities.`,
    maskedOutput: `🛡️ VERIFIED EVIDENCE  ⏱️ 0.48 ms

Hello! I am NOC Intelligence, an AI-powered Network Operations Center assistant. You can ask me about trouble ticket statuses (INC-2025-XXXXX), work order resolutions (WO-70XXXX), leased circuits (CKT-88XXXX), customer ticket counts, or technical network symptoms.

🔍 Safe Telemetry:
Route Type: CLARIFICATION | Intent: CLARIFICATION_GREETING | Language: EN`
  },

  greeting_th: {
    query: "สวัสดีครับ ช่วยอะไรได้บ้าง",
    route: "CLARIFICATION",
    intent: "CLARIFICATION_GREETING",
    target: "Direct Conversational Capabilities Guide (Zero DB execution)",
    latency: "0.45 ms (E2E)",
    status: "VERIFIED",
    badgeClass: "score-success",
    evidence: `[ROUTER PRECEDENCE RULE 8b: RE_GREETING MATCH]
Thai language greeting detected.
Target: Guided clarification prompt in Thai.`,
    maskedOutput: `🛡️ VERIFIED EVIDENCE  ⏱️ 0.45 ms

สวัสดีครับ ผมคือผู้ช่วยอัจฉริยะศูนย์ปฏิบัติการเครือข่าย (NOC Intelligence) คุณสามารถสอบถามข้อมูลสถานะตั๋วปัญหา (INC-2025-XXXXX), รายละเอียดการซ่อมใบงานช่าง (WO-70XXXX), ข้อมูลวงจร (CKT-88XXXX), ยอดตั๋วลูกค้า หรือค้นหาสาเหตุและอาการขัดข้องของระบบเครือข่ายได้ครับ

🔍 Safe Telemetry:
Route Type: CLARIFICATION | Intent: CLARIFICATION_GREETING | Language: TH`
  },

  outofdomain: {
    query: "พยากรณ์อากาศที่กรุงเทพวันนี้ฝนจะตกไหม",
    route: "UNSUPPORTED",
    intent: "UNSUPPORTED_OUT_OF_SCOPE",
    target: "Domain Boundary Gate (Strictly rejected)",
    latency: "0.57 ms (E2E)",
    status: "UNSUPPORTED",
    badgeClass: "score-warning",
    evidence: `[ROUTER PRECEDENCE RULE 9: FALLBACK / OUT-OF-DOMAIN]
Query does not match operational identifiers, symptoms, or greetings.
Action: Safe domain boundary rejection.`,
    maskedOutput: `🛡️ OUT-OF-DOMAIN REJECTION  ⏱️ 0.57 ms

ระบบ NOC Intelligence รองรับเฉพาะข้อมูลปฏิบัติการโทรคมนาคม (ตั๋วปัญหา ใบงาน วงจร และลูกค้า) เท่านั้นครับ

⚠️ Query rejected by deterministic routing.

🔍 Safe Telemetry:
Route Type: UNSUPPORTED | Intent: UNSUPPORTED_OUT_OF_SCOPE | Status: UNSUPPORTED`
  }
};

function initSimulator() {
  const buttons = document.querySelectorAll('.sim-btn');
  const queryDisplay = document.getElementById('simQuery');
  const routeDisplay = document.getElementById('simRoute');
  const intentDisplay = document.getElementById('simIntent');
  const targetDisplay = document.getElementById('simTarget');
  const latencyDisplay = document.getElementById('simLatency');
  const statusDisplay = document.getElementById('simStatus');
  const evidenceDisplay = document.getElementById('simEvidence');
  const outputDisplay = document.getElementById('simOutput');

  if (!buttons.length || !queryDisplay) return;

  const loadScenario = (key) => {
    const sc = SIM_SCENARIOS[key];
    if (!sc) return;

    buttons.forEach(b => b.classList.toggle('active', b.dataset.scenario === key));

    queryDisplay.innerText = sc.query;
    routeDisplay.innerText = sc.route;
    intentDisplay.innerText = sc.intent;
    targetDisplay.innerText = sc.target;
    latencyDisplay.innerText = sc.latency;
    statusDisplay.innerText = sc.status;
    statusDisplay.className = `score-tag ${sc.badgeClass}`;
    evidenceDisplay.innerText = sc.evidence;
    outputDisplay.innerText = sc.maskedOutput;
  };

  buttons.forEach(btn => {
    btn.addEventListener('click', () => {
      loadScenario(btn.dataset.scenario);
    });
  });

  // Default load first scenario
  loadScenario('ticket');
}

// ─── 3. Copy Code Buttons ───────────────────────────────────────────────────
function initCopyButtons() {
  const copyBtns = document.querySelectorAll('.copy-btn');
  copyBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      const textToCopy = btn.dataset.copy;
      if (textToCopy) {
        navigator.clipboard.writeText(textToCopy).then(() => {
          const original = btn.innerText;
          btn.innerText = 'Copied!';
          btn.style.borderColor = '#10b981';
          btn.style.color = '#10b981';
          setTimeout(() => {
            btn.innerText = original;
            btn.style.borderColor = '';
            btn.style.color = '';
          }, 2000);
        });
      }
    });
  });
}

// ─── 4. Mobile Navigation ───────────────────────────────────────────────────
function initMobileNav() {
  const toggle = document.querySelector('.mobile-toggle');
  const menu = document.querySelector('.nav-menu');

  if (toggle && menu) {
    toggle.addEventListener('click', () => {
      menu.classList.toggle('mobile-open');
    });

    document.querySelectorAll('.nav-link').forEach(link => {
      link.addEventListener('click', () => {
        menu.classList.remove('mobile-open');
      });
    });
  }
}

// ─── 5. ScrollSpy Active Link Tracking ──────────────────────────────────────
function initScrollSpy() {
  const sections = document.querySelectorAll('section[id]');
  const navLinks = document.querySelectorAll('.nav-link');

  window.addEventListener('scroll', () => {
    let current = '';
    sections.forEach(section => {
      const sectionTop = section.offsetTop - 120;
      if (window.scrollY >= sectionTop) {
        current = section.getAttribute('id');
      }
    });

    navLinks.forEach(link => {
      link.classList.remove('active');
      if (link.getAttribute('href') === `#${current}`) {
        link.classList.add('active');
      }
    });
  });
}
