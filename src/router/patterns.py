"""
NOC Intelligence — Stage 4 Deterministic Patterns & Regex Engine
Precompiled regular expressions, entity boundaries, and keyword dictionaries.
"""

import re

# ─── 1. Canonical Synthetic Identifiers ────────────────────────────────────────

# Matches trouble tickets: INC-2025-00001 through INC-2025-02500
RE_TICKET_ID = re.compile(r"\bINC-2025-\d{5}\b", re.IGNORECASE)

# Matches engineering work orders: WO-700001 through WO-703000
RE_WORK_ORDER_ID = re.compile(r"\bWO-70\d{4}\b", re.IGNORECASE)

# Matches telecommunications circuits: CKT-880001 through CKT-880300
RE_CIRCUIT_REF = re.compile(r"\bCKT-88\d{4}\b", re.IGNORECASE)

# Matches customer codes: CUST-001 through CUST-050
RE_CUSTOMER_CODE = re.compile(r"\bCUST-\d{3}\b", re.IGNORECASE)

# Detects potential malicious SQL injections / escape sequences
RE_SQL_INJECTION = re.compile(
    r"(\b(union\s+select|select\s+.*\s+from|insert\s+into|drop\s+table|delete\s+from|alter\s+table|exec\s*\(|execute\s*\(|xp_cmdshell)\b|--\s*|;\s*(drop|select|insert|delete|alter)|\b1\s*=\s*1\b|\bor\s+true\b)",
    re.IGNORECASE
)


# ─── 2. Intent & Traversal Regex & Keyword Patterns ────────────────────────────

# Regex for Bridge traversal (Ticket -> Work Orders)
RE_BRIDGE_TICKET_TO_WO = re.compile(
    r"(\blink(ed)?\s+.*work\s*order|work\s*order\s+.*link(ed)?|associated\s+work\s*order|field\s+work\s*order|ใบงาน.*ไหน|เชื่อมโยง.*ใบงาน|ผูกกับใบงาน|มีงานซ่อมอะไรบ้าง)",
    re.IGNORECASE
)

# Regex for Reverse bridge traversal (Work Order -> Tickets)
RE_BRIDGE_WO_TO_TICKETS = re.compile(
    r"((affect|impact)(ed)?\s+.*ticket|ticket.*(affect|impact)(ed)?|linked\s+ticket|กระทบ.*ตั๋ว|ตั๋ว.*กระทบ|ตั๋ว.*ใบไหน|ผูกกับตั๋ว)",
    re.IGNORECASE
)

# Keywords indicating ticket counting / aggregation
COUNT_KEYWORDS = {
    "กี่รายการ", "กี่ใบ", "มีตั๋วทั้งหมดกี่", "จำนวนตั๋ว", "ยอดตั๋ว", "กี่เคส",
    "ยอดรวมตั๋ว", "ยอดรวมตั๋วปัญหา", "ยอดตั๋วปัญหา",
    "how many tickets", "ticket count", "total tickets", "number of tickets",
    "how many incidents", "count tickets"
}

# Keywords indicating circuit inventory
CIRCUIT_INVENTORY_KEYWORDS = {
    "วงจร", "มีวงจรอะไรบ้าง", "วงจรทั้งหมด", "รายการวงจร", "circuit list",
    "circuits for", "customer circuits", "leased line", "leased lines", "ขอรายการวงจร",
    "circuits does", "circuits registered"
}

# Keywords indicating full incident history / chronological activities
HISTORY_KEYWORDS = {
    "ประวัติ", "timeline", "กิจกรรม", "activity log", "activities", "history",
    "ความคืบหน้า", "ขั้นตอนการซ่อม", "บันทึกช่าง"
}

# Normalized status filters
STATUS_KEYWORDS_MAP = {
    "OPEN": "OPEN",
    "เปิด": "OPEN",
    "ค้าง": "OPEN",
    "IN_PROGRESS": "IN_PROGRESS",
    "กำลังดำเนินการ": "IN_PROGRESS",
    "กำลังทำ": "IN_PROGRESS",
    "RESOLVED": "RESOLVED",
    "แก้ไขแล้ว": "RESOLVED",
    "แก้แล้ว": "RESOLVED",
    "CLOSED": "CLOSED",
    "ปิดงานแล้ว": "CLOSED",
    "ปิดแล้ว": "CLOSED"
}

# Keywords indicating semantic symptoms, technical failures, or root causes
SEMANTIC_SYMPTOM_KEYWORDS = [
    # Fiber / Optical
    "fiber cut", "fibre cut", "สายขาด", "สายไฟเบอร์", "สายเคเบิล", "สายสัญญาณ", "ใยแก้ว", "ตัดสาย",
    "optical degradation", "fiber degradation", "degradation", "rx level", "rx power",
    "ค่าแสงตก", "loss of signal", "los alarm", "sfp", "sfp+", "transceiver", "โมดูลแสง",
    "หัวรับส่งสัญญาณแสง",
    # Power
    "power outage", "commercial power", "ไฟดับ", "หม้อแปลงระเบิด", "pea", "การไฟฟ้า",
    "ups", "battery depleted", "แบตเตอรี่หมด",
    # Routing / Hardware / Traffic Congestion
    "congestion", "ddos", "traffic congestion", "traffic spike", "bandwidth saturation",
    "bgp", "route flap", "flapping", "flap", "ทราฟฟิกตก", "ปริมาณทราฟฟิกสูง", "packet loss",
    "แพ็กเก็ตสูญหาย", "latency spike", "ค่าความหน่วง", "jitter", "jumbo frame",
    "mtu mismatch", "mtu", "memory parity fault", "device reload", "รีบูต",
    "hardware failure", "สวิตช์เสีย", "พาวเวอร์ซัพพลาย", "psu",
    # General Root Cause / Resolution questions
    "สาเหตุคืออะไร", "เกิดจากสาเหตุใด", "ทำไมถึงเสีย", "แก้ไขอย่างไร", "ช่างแก้ไขอะไร",
    "ซ่อมแซมอย่างไร", "วิธีแก้ไขปัญหา", "วิธีแก้", "root cause", "specific cause",
    "action taken", "remediation", "resolve"
]

# Regex for vague complaints
RE_VAGUE_COMPLAINT = re.compile(
    r"(internet.*slow|slow.*internet|connection.*(down|drop)|not\s+working|extremely\s+slow|เน็ตช้า|เน็ตหลุด|ใช้งานไม่ได้|ช้ามาก|ช่วยดูให้หน่อย|ช่วยตรวจสอบที|ดูระบบให้หน่อย)",
    re.IGNORECASE
)

# Regex for status inquiry without ID
RE_STATUS_INQUIRY = re.compile(
    r"(สถานะเป็นอะไร|สถานะอะไร|เสร็จหรือยัง|ปิดงานหรือยัง|เป็นอย่างไรบ้าง|what\s+is\s+the\s+status|check\s+status|is\s+it\s+resolved|ticket\s+status)",
    re.IGNORECASE
)

# Regex for conversational greetings and capability questions
RE_GREETING = re.compile(
    r"\b(hello|hi|hey|greetings|good\s+(morning|afternoon|evening)|help|what\s+can\s+you\s+do|who\s+are\s+you|how\s+can\s+you\s+help)\b|"
    r"(สวัสดี|หวัดดี|ช่วยอะไรได้บ้าง|ทำอะไรได้บ้าง|คุณคือใคร|แนะนำตัว)",
    re.IGNORECASE
)
