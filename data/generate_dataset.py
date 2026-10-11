#!/usr/bin/env python3
"""
NOC Intelligence — Synthetic Telecom / NOC Dataset Generator
Stage 1: Generates 15,050 relational synthetic records across 6 entity tables:
- 50 Customers
- 300 Circuits
- 2,500 Trouble Tickets
- 3,000 Work Orders
- 3,200 Ticket-Work-Order Bridge Links
- 6,000 Work Order Activities

100% Fictional Data. Zero proprietary/company assets. Fully deterministic via seed.
"""

import os
import sys
import json
import csv
import random
from datetime import datetime, timedelta

DEFAULT_SEED = 42

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "synthetic")
FIXTURES_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ground_truth_fixtures.json")


# ─── Fictional Master Taxonomies ──────────────────────────────────────────────

FICTIONAL_COMPANIES = [
    ("Apex Logistics", "บริษัท เอเพ็กซ์ โลจิสติกส์ จำกัด", "Logistics", "Platinum (99.99%)"),
    ("Horizon Retail Group", "บริษัท ฮอไรซอน รีเทล กรุ๊ป จำกัด (มหาชน)", "Retail", "Gold (99.95%)"),
    ("Nova Cloud Solutions", "บริษัท โนวา คลาวด์ โซลูชันส์ จำกัด", "Cloud & Data Center", "Platinum (99.99%)"),
    ("Zenith Energy Corp", "บริษัท ซีนิท เอนเนอร์ยี่ คอร์ปอเรชั่น จำกัด (มหาชน)", "Energy & Utilities", "Platinum (99.99%)"),
    ("Solaria Health Systems", "บริษัท โซลาเรีย เฮลท์ ซิสเต็มส์ จำกัด", "Healthcare", "Platinum (99.99%)"),
    ("Meridian Agro Industries", "บริษัท เมอริเดียน อุตสาหกรรมเกษตร จำกัด", "Manufacturing", "Gold (99.95%)"),
    ("Quantum Financial Services", "บริษัท ควอนตัม ไฟแนนเชียล เซอร์วิสเซส จำกัด", "Finance & Banking", "Platinum (99.99%)"),
    ("Vertex Media Networks", "บริษัท เวอร์เท็กซ์ มีเดีย เน็ตเวิร์กส์ จำกัด", "Telecommunications", "Gold (99.95%)"),
    ("Titan Robotics & Automation", "บริษัท ไททัน โรโบติกส์ แอนด์ ออโตเมชัน จำกัด", "Manufacturing", "Gold (99.95%)"),
    ("Atlas Trans-Logistics", "บริษัท แอตลาส ทรานส์-โลจิสติกส์ จำกัด", "Logistics", "Silver (99.90%)"),
    ("Orion Semiconductor", "บริษัท โอไรออน เซมิคอนดักเตอร์ จำกัด", "Manufacturing", "Platinum (99.99%)"),
    ("BlueWave Hospitality", "บริษัท บลูเวฟ ฮอสพิทอลลิตี้ กรุ๊ป จำกัด (มหาชน)", "Retail", "Silver (99.90%)"),
    ("Pinnacle Construction", "บริษัท พินนาเคิล คอนสตรัคชั่น จำกัด", "Manufacturing", "Silver (99.90%)"),
    ("Vanguard Insurance", "บริษัท แวนการ์ด ประกันภัย จำกัด (มหาชน)", "Finance & Banking", "Gold (99.95%)"),
    ("Solartech Clean Energy", "บริษัท โซลาร์เทค พลังงานสะอาด จำกัด", "Energy & Utilities", "Gold (99.95%)"),
    ("Nexa E-Commerce Platform", "บริษัท เน็กซา อีคอมเมิร์ซ แพลตฟอร์ม จำกัด", "Retail", "Platinum (99.99%)"),
    ("Crestwood Paper & Packaging", "บริษัท เครสต์วูด กระดาษและบรรจุภัณฑ์ จำกัด", "Manufacturing", "Silver (99.90%)"),
    ("Silverline Pharma", "บริษัท ซิลเวอร์ไลน์ ฟาร์มาซูติคอล จำกัด", "Healthcare", "Gold (99.95%)"),
    ("Metro Rail Systems", "บริษัท เมโทร เรล ซิสเต็มส์ จำกัด (มหาชน)", "Transportation", "Platinum (99.99%)"),
    ("Omni Consumer Goods", "บริษัท ออมนิ คอนซูเมอร์ กู้ดส์ จำกัด", "Retail", "Gold (99.95%)"),
    ("CyberShield Defense", "บริษัท ไซเบอร์ชิลด์ ซีเคียวริตี้ จำกัด", "Cloud & Data Center", "Platinum (99.99%)"),
    ("Emerald Petrochemicals", "บริษัท เอเมอรัลด์ ปิโตรเคมิคอล จำกัด (มหาชน)", "Energy & Utilities", "Platinum (99.99%)"),
    ("Aurora Aviation Services", "บริษัท ออโรร่า เอวิเอชั่น เซอร์วิสเซส จำกัด", "Transportation", "Gold (99.95%)"),
    ("Golden Harvest Foods", "บริษัท โกลเด้น ฮาร์เวสต์ ฟู้ดส์ จำกัด", "Manufacturing", "Silver (99.90%)"),
    ("Trinity Smart Grid", "บริษัท ทรินิตี้ สมาร์ท กริด จำกัด", "Energy & Utilities", "Platinum (99.99%)"),
    ("Pacific Express Courier", "บริษัท แปซิฟิก เอ็กซ์เพรส คูเรียร์ จำกัด", "Logistics", "Gold (99.95%)"),
    ("Sterling Asset Management", "บริษัท สเตอร์ลิง แอสเสท แมเนจเมนท์ จำกัด", "Finance & Banking", "Platinum (99.99%)"),
    ("Hyperion Data Centers", "บริษัท ไฮเปอเรียน ดาต้า เซ็นเตอร์ จำกัด", "Cloud & Data Center", "Platinum (99.99%)"),
    ("Falcon Automotive Parts", "บริษัท ฟอลคอน ชิ้นส่วนยานยนต์ จำกัด", "Manufacturing", "Silver (99.90%)"),
    ("Siam Nano Materials", "บริษัท สยาม นาโน แมททีเรียลส์ จำกัด", "Manufacturing", "Gold (99.95%)"),
    ("Kaset Phatthana Biofuel", "บริษัท เกษตรพัฒนา ไบโอฟูเอล จำกัด (มหาชน)", "Energy & Utilities", "Silver (99.90%)"),
    ("Krungthep Smart City Solutions", "บริษัท กรุงเทพ สมาร์ท ซิตี้ โซลูชั่นส์ จำกัด", "Telecommunications", "Gold (99.95%)"),
    ("Thai Precision Instruments", "บริษัท ไทย พรีซิชั่น อินสตรูเมนท์ส จำกัด", "Manufacturing", "Silver (99.90%)"),
    ("Lanna Textile Mills", "บริษัท ลานนา เท็กซ์ไทล์ มิลส์ จำกัด", "Manufacturing", "Silver (99.90%)"),
    ("Andaman Maritime Shipping", "บริษัท อันดามัน มาริไทม์ ชิปปิ้ง จำกัด", "Transportation", "Gold (99.95%)"),
    ("Chonburi Cold Storage", "บริษัท ชลบุรี โคลด์ สตอเรจ จำกัด", "Logistics", "Gold (99.95%)"),
    ("Isan Renewable Power", "บริษัท อีสาน พลังงานหมุนเวียน จำกัด", "Energy & Utilities", "Silver (99.90%)"),
    ("Eastern Seaboard Logistics", "บริษัท อีสเทิร์น ซีบอร์ด โลจิสติกส์ จำกัด", "Logistics", "Platinum (99.99%)"),
    ("Ayutthaya Microelectronics", "บริษัท อยุธยา ไมโครอิเล็กทรอนิกส์ จำกัด", "Manufacturing", "Platinum (99.99%)"),
    ("Thonburi Heavy Machinery", "บริษัท ธนบุรี เครื่องจักรกลหนัก จำกัด", "Manufacturing", "Silver (99.90%)"),
    ("Rayong Chemical Polymers", "บริษัท ระยอง เคมีคอล โพลิเมอร์ส จำกัด", "Manufacturing", "Gold (99.95%)"),
    ("Saraburi Cement Tech", "บริษัท สระบุรี ซีเมนต์ เทค จำกัด", "Manufacturing", "Silver (99.90%)"),
    ("Samut Prakan Port Terminal", "บริษัท สมุทรปราการ พอร์ต เทอร์มินอล จำกัด", "Transportation", "Platinum (99.99%)"),
    ("Nonthaburi Telecom Infra", "บริษัท นนทบุรี เทเลคอม อินฟรา จำกัด", "Telecommunications", "Platinum (99.99%)"),
    ("Pathum Thani Biotech Lab", "บริษัท ปทุมธานี ไบโอเทค แล็บ จำกัด", "Healthcare", "Gold (99.95%)"),
    ("Korat Solar Farm", "บริษัท โคราช โซลาร์ ฟาร์ม จำกัด", "Energy & Utilities", "Silver (99.90%)"),
    ("Phuket Resort Networks", "บริษัท ภูเก็ต รีสอร์ท เน็ตเวิร์กส์ จำกัด", "Retail", "Silver (99.90%)"),
    ("Songkhla Marine Logistics", "บริษัท สงขลา มารีน โลจิสติกส์ จำกัด", "Logistics", "Gold (99.95%)"),
    ("Phitsanulok Grain Storage", "บริษัท พิษณุโลก เกรน สตอเรจ จำกัด", "Logistics", "Silver (99.90%)"),
    ("Ubon Green Hydroponics", "บริษัท อุบล กรีน ไฮโดรโปนิกส์ จำกัด", "Manufacturing", "Silver (99.90%)")
]

SITE_REGIONS = [
    ("Bangkok Silom Head Office", "BKK-01", "Central"),
    ("Bangkok Sukhumvit Branch", "BKK-02", "Central"),
    ("Bangkok Sathorn Finance Hub", "BKK-03", "Central"),
    ("Samut Prakan Distribution Hub", "SPK-01", "Central"),
    ("Nonthaburi Operations Site", "NTB-01", "Central"),
    ("Pathum Thani Tech Campus", "PTE-01", "Central"),
    ("Ayutthaya Industrial Depot", "AYA-01", "Central"),
    ("Chonburi Eastern Terminal", "CBI-01", "East"),
    ("Rayong Petro Plant Branch", "RYG-01", "East"),
    ("Chachoengsao Logistics Base", "CCO-01", "East"),
    ("Chiang Mai North Gateway", "CNX-01", "North"),
    ("Chiang Rai Regional Office", "CEI-01", "North"),
    ("Phitsanulok Transit Node", "PHS-01", "North"),
    ("Lampang Distribution Yard", "LPG-01", "North"),
    ("Nakhon Ratchasima Hub", "NAK-01", "Northeast"),
    ("Khon Kaen Regional Depot", "KKN-01", "Northeast"),
    ("Ubon Ratchathani Site", "UBN-01", "Northeast"),
    ("Udon Thani Commercial Center", "UDN-01", "Northeast"),
    ("Songkhla Southern Gateway", "SKA-01", "South"),
    ("Hat Yai Central Switch", "HYI-01", "South"),
    ("Phuket Island Data Link", "HKT-01", "South"),
    ("Surat Thani Terminal", "URT-01", "South"),
    ("Krabi Coastal Branch", "KBV-01", "South"),
    ("Saraburi Logistics Center", "SRB-01", "Central")
]

INTERFACE_TYPES = [
    ("FTTx Dedicated Fiber", 100),
    ("FTTx Dedicated Fiber", 500),
    ("MPLS L3VPN Metro", 200),
    ("MPLS L3VPN Metro", 1000),
    ("Dark Fiber Metro Link", 10000),
    ("Dark Fiber Metro Link", 2000),
    ("Microwave Radio Backup", 100),
    ("Metro Ethernet Direct", 1000)
]

SYMPTOM_TEMPLATES = [
    ("LINK_DOWN", "P1-Critical", "Complete link loss / สัญญาณขาดหายทั้งหมด ไม่สามารถเชื่อมต่อเกตเวย์ได้"),
    ("LINK_DOWN", "P1-Critical", "Optical port link down alarm on interface GigabitEthernet0/0/1 / พอร์ตสัญญาณดาวน์"),
    ("PACKET_LOSS", "P2-Major", "Intermittent packet loss >40% during peak traffic / แพ็กเก็ตสูญหายสูงกว่า 40%"),
    ("PACKET_LOSS", "P2-Major", "Burst frame drops observed on core uplink / ตรวจพบเฟรมหลุดต่อเนื่องที่อัปลิงก์"),
    ("LATENCY_SPIKE", "P3-Minor", "Latency spike >180ms to core switch / ค่าความหน่วงสูงผิดปกติเกิน 180ms"),
    ("LATENCY_SPIKE", "P3-Minor", "Jitter degradation impacting VoIP sessions / ค่าความแปรปรวนสัญญาณกระทบระบบโทรศัพท์"),
    ("OPTICAL_DEGRADATION", "P2-Major", "Optical RX level degraded to -28.4 dBm (Threshold: -24 dBm) / ค่าแสงขาต่ำกว่าเกณฑ์วิกฤต"),
    ("OPTICAL_DEGRADATION", "P3-Minor", "Optical receiver power fluctuation warning / ตรวจพบค่ากำลังแสงสวิงต่อเนื่อง"),
    ("BGP_FLAP", "P2-Major", "BGP peering state flapping every 120 seconds / เซสชัน BGP หลุดและต่อใหม่ทุก 2 นาที"),
    ("BGP_FLAP", "P3-Minor", "Route flapping and prefix count fluctuation / เกิดการกระพริบของเส้นทางเครือข่าย"),
    ("EQUIPMENT_POWER", "P1-Critical", "Dual power supply failure alarm received from CPE / แจ้งเตือนระบบไฟของอุปกรณ์ขัดข้อง"),
    ("EQUIPMENT_POWER", "P2-Major", "High temperature warning on aggregation switch / อุณหภูมิอุปกรณ์รวมสายสูงเกินพิกัด")
]

WORK_ORDER_CAUSES = [
    (
        "FIBER_CUT",
        "สายไฟเบอร์ออปติกภายนอกขาดชำรุด",
        "Third-party construction road excavation severed 48-core OSP fiber cable at KM 14.2 / ผู้รับเหมาขุดเจาะถนนตัดสายเคเบิลขาด",
        "OSP splicing crew dispatched. Performed fusion splicing on 48 cores, OTDR tested within normal optical limits / ช่างเข้าตัดต่อสายเคเบิลและวัดค่าแสง OTDR ผ่านเกณฑ์",
        "Fiber Patch Panel",
        "FDF-Enclosure-144C",
        "Transmission OSP Splicing Team"
    ),
    (
        "OPTICAL_TRANSCEIVER",
        "หัวรับส่งสัญญาณแสง (SFP+) เสื่อมสภาพ",
        "SFP+ 10G-LR optical module laser bias degradation causing severe bit errors / โมดูลแสงเลเซอร์เสื่อมสภาพทำให้เกิดสัญญาณรบกวน",
        "Replaced faulty SFP-10G-LR optical transceiver with new spare unit. Cleaned LC patch cord connector / เปลี่ยนโมดูลแสง SFP ตัวใหม่และทำความสะอาดหัวคอนเน็กเตอร์",
        "SFP+ 10G Transceiver",
        "Module-LR-10G",
        "Metro Field Engineering"
    ),
    (
        "COMMERCIAL_POWER",
        "ไฟฟ้าดับที่สถานีหรืออาคารลูกค้า",
        "Regional commercial power blackout; building backup UPS batteries depleted after 4 hours / ไฟฟ้าดับบริเวณกว้าง แบตเตอรี่สำรองใช้งานจนหมด",
        "Coordinated with facility team. Restored main circuit breaker after municipal electricity stabilized / ประสานเจ้าหน้าที่อาคารสับสวิตช์เบรกเกอร์หลักเมื่อไฟฟ้าหลวงกลับมาปกติ",
        "Rectifier / UPS Unit",
        "UPS-Modular-5KVA",
        "Power & Facilities Support"
    ),
    (
        "HARDWARE_FAILURE",
        "การ์ดประมวลผลหรือสวิตช์เครือข่ายชำรุด",
        "Main routing processor board memory parity fault causing intermittent device reload / แผงวงจรควบคุมหลักพบข้อผิดพลาดหน่วยความจำ ส่งผลให้เครื่องรีบูต",
        "Hot-swapped backup route processor module (RP-2). Upgraded device microcode to stable patch / สลับใช้งานการ์ดประมวลผลสำรองและอัปเกรดเฟิร์มแวร์เวอร์ชันเสถียร",
        "Core Switch",
        "Model-CR9000-Core",
        "Tier-2 IP Core NOC"
    ),
    (
        "CONFIGURATION_ERROR",
        "การกำหนดค่าคอนฟิกูเรชันผิดพลาด",
        "Mismatched MTU setting on sub-interface and route-map filter dropped customer Jumbo frames / ขนาด MTU ไม่ตรงกันและฟิลเตอร์ตัดแพ็กเก็ตขนาดใหญ่",
        "Corrected MTU configuration to 9216 on PE sub-interface; verified BGP route-policy filters / ปรับแก้ค่า MTU ให้ถูกต้องและตรวจสอบนโยบายการกระจายเส้นทาง",
        "Edge Router",
        "Router-PE-Aggregation",
        "Tier-2 IP Core NOC"
    ),
    (
        "CONGESTION",
        "ปริมาณทราฟฟิกหนาแน่นเกินพิกัดพอร์ต",
        "Customer unexpected backup replication burst congested 1Gbps uplink interface / ปริมาณการสำรองข้อมูลกระชากสูงเกินพิกัดพอร์ต 1Gbps",
        "Temporarily tuned QoS prioritization queue; coordinated customer bandwidth upgrade / ปรับแต่งคิวจัดลำดับความสำคัญของแพ็กเก็ต (QoS) และแนะนำขยายช่องสัญญาณ",
        "Edge Router",
        "Router-PE-Aggregation",
        "Tier-2 IP Core NOC"
    ),
    (
        "ENVIRONMENTAL",
        "ความร้อนหรือน้ำท่วมขังในตู้จุดเชื่อมต่อ",
        "Heavy rainstorm caused water seepage into underground handhole closure / น้ำฝนซึมเข้าตู้พักสายใต้ดินส่งผลให้คอนเน็กเตอร์ชื้น",
        "Drained manhole, cleaned and heat-shrink resealed waterproof splice enclosure / สูบน้ำระบายความชื้น ทำความสะอาดและซีลกันน้ำปลอกหุ้มข้อต่อใหม่",
        "Fiber Patch Panel",
        "Dome-Closure-Waterproof",
        "Transmission OSP Splicing Team"
    )
]

ACTIVITY_SEQUENCES = [
    ("DISPATCHED", "NOC dispatched field response unit to investigated site / ศูนย์ NOC ได้สั่งการทีมช่างภาคสนามเข้าพื้นที่"),
    ("EN_ROUTE", "Technician en-route to customer site; estimated arrival in 40 minutes / ช่างกำลังเดินทาง คาดว่าจะถึงหน้างานภายใน 40 นาที"),
    ("ON_SITE_DIAGNOSIS", "On-site optical power testing: RX reads below nominal threshold / ช่างถึงจุดเกิดเหตุ วัดค่ากำลังแสงพบต่ำกว่าเกณฑ์"),
    ("CORRECTIVE_ACTION", "Performed physical remediation and component replacement / ดำเนินการแก้ไขทางกายภาพและเปลี่ยนอุปกรณ์ทดแทน"),
    ("SERVICE_VERIFICATION", "Ping loopback and bit-error testing passed cleanly with zero packet loss / ทดสอบสัญญาณลูปแบ็กผ่านฉลุย ไม่พบแพ็กเก็ตสูญหาย"),
    ("CLOSEOUT", "Customer technical contact confirmed service recovery. Work order closed / ผู้ประสานงานลูกค้ายืนยันระบบกลับมาใช้งานได้ตามปกติ ปิดงานสมบูรณ์")
]


# ─── Data Generator Engine ───────────────────────────────────────────────────

def generate_all_datasets(seed=DEFAULT_SEED):
    random.seed(seed)
    
    print(f"[*] Initializing Synthetic NOC Dataset Generator (seed={seed})...")
    
    # 1. Customers (50 rows)
    customers = []
    for cid in range(1, 51):
        name_en, name_th, industry, sla = FICTIONAL_COMPANIES[cid - 1]
        customers.append({
            "customer_id": cid,
            "customer_code": f"CUST-{cid:03d}",
            "company_name_en": name_en,
            "company_name_th": name_th,
            "company_short_name": name_en.split()[0],
            "industry_segment": industry,
            "sla_tier": sla,
            "account_status": "Active",
            "created_at": "2024-01-15T08:00:00Z"
        })
    print(f"  [+] Generated {len(customers)} Customers")

    # 2. Circuits (300 rows)
    circuits = []
    circuit_id = 1
    # Distribute 300 circuits among 50 customers (6 circuits per customer on average)
    for cid in range(1, 51):
        num_circuits = 6  # 50 * 6 = 300
        for idx in range(num_circuits):
            site_name, pop_code, region = SITE_REGIONS[(cid * 3 + idx) % len(SITE_REGIONS)]
            iface, bw = INTERFACE_TYPES[(circuit_id + idx) % len(INTERFACE_TYPES)]
            cust_name = customers[cid - 1]["company_short_name"]
            circuits.append({
                "circuit_id": circuit_id,
                "circuit_ref": f"CKT-88{circuit_id:04d}",
                "customer_id": cid,
                "site_name": f"{cust_name} {site_name}",
                "region": region,
                "bandwidth_mbps": bw,
                "interface_type": iface,
                "gateway_pop": f"POP-{pop_code}",
                "status": "ACTIVE",
                "created_at": "2024-06-01T00:00:00Z"
            })
            circuit_id += 1
    print(f"  [+] Generated {len(circuits)} Circuits")

    # Lookup helpers
    circuits_by_customer = {}
    for c in circuits:
        circuits_by_customer.setdefault(c["customer_id"], []).append(c)

    # 3. Trouble Tickets (2,500 rows)
    # Timestamps span Jan 1, 2025 to Mar 25, 2025
    base_time = datetime(2025, 1, 1, 0, 0, 0)
    tickets = []
    
    for tid in range(1, 2501):
        cid = (tid % 50) + 1
        cust_circuits = circuits_by_customer[cid]
        circuit = cust_circuits[tid % len(cust_circuits)]
        
        # Deterministic open time
        day_offset = (tid * 83) % 84  # 84 days
        minute_offset = (tid * 47) % 1440
        dt_open = base_time + timedelta(days=day_offset, minutes=minute_offset)
        
        # Status distribution: 92% Closed/Resolved, 8% In Progress
        is_in_progress = (tid % 13 == 0)
        status = "IN_PROGRESS" if is_in_progress else ("CLOSED" if (tid % 2 == 0) else "RESOLVED")
        
        symptom_cat, severity, symptom_desc = SYMPTOM_TEMPLATES[tid % len(SYMPTOM_TEMPLATES)]
        
        if is_in_progress:
            dt_close = None
        else:
            duration_mins = 20 + ((tid * 17) % 480)  # 20 min to 8.3 hrs
            dt_close = dt_open + timedelta(minutes=duration_mins)
            
        tickets.append({
            "ticket_id": f"INC-2025-{tid:05d}",
            "customer_id": cid,
            "circuit_id": circuit["circuit_id"],
            "circuit_ref": circuit["circuit_ref"],
            "severity": severity,
            "status": status,
            "symptom_category": symptom_cat,
            "reported_symptom": symptom_desc,
            "contact_person": f"Contact_Person_{cid}_{tid%5 + 1}",
            "contact_phone": f"+66-2-000-{tid%9000 + 1000:04d}",
            "dt_opened": dt_open.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "dt_closed": dt_close.strftime("%Y-%m-%dT%H:%M:%SZ") if dt_close else ""
        })
    print(f"  [+] Generated {len(tickets)} Trouble Tickets")

    # 4. Work Orders (3,000 rows)
    # Work orders span the same time horizon; will be correlated with tickets
    work_orders = []
    for wid in range(1, 3001):
        # Tie dispatch time around incidents
        t_ref = tickets[(wid * 7) % len(tickets)]
        t_open = datetime.strptime(t_ref["dt_opened"], "%Y-%m-%dT%H:%M:%SZ")
        
        dt_disp = t_open + timedelta(minutes=10 + ((wid * 11) % 40))
        
        is_in_progress = (wid % 17 == 0)
        is_cancelled = (wid % 43 == 0 and not is_in_progress)
        
        if is_in_progress:
            wo_status = "IN_PROGRESS"
            dt_rest = None
            dt_close = None
        elif is_cancelled:
            wo_status = "CANCELLED"
            dt_rest = None
            dt_close = dt_disp + timedelta(minutes=25)
        else:
            wo_status = "CLOSED" if wid % 2 == 0 else "COMPLETED"
            dt_rest = dt_disp + timedelta(minutes=30 + ((wid * 23) % 240))
            dt_close = dt_rest + timedelta(minutes=15 + ((wid * 7) % 60))
            
        cause_data = WORK_ORDER_CAUSES[wid % len(WORK_ORDER_CAUSES)]
        
        work_orders.append({
            "work_order_id": f"WO-70{wid:04d}",
            "status": wo_status,
            "root_cause_category": cause_data[0],
            "root_cause_category_th": cause_data[1],
            "specific_cause": cause_data[2],
            "action_taken": cause_data[3],
            "equipment_type": cause_data[4],
            "equipment_model": cause_data[5],
            "assigned_team": cause_data[6],
            "field_engineer_alias": f"Engineer_Tech_{(wid % 40) + 1:02d}",
            "dt_dispatched": dt_disp.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "dt_restored": dt_rest.strftime("%Y-%m-%dT%H:%M:%SZ") if dt_rest else "",
            "dt_closed": dt_close.strftime("%Y-%m-%dT%H:%M:%SZ") if dt_close else ""
        })
    print(f"  [+] Generated {len(work_orders)} Work Orders")

    # 5. Ticket-Work-Order Bridge Links (3,200 rows)
    # 2,500 1-to-1 links (for all tickets) + 700 additional links to demonstrate 1:M and M:N
    links = []
    link_set = set()
    link_id = 1
    
    # Pass 1: Link each ticket to at least one work order (2,500 links)
    for tid_idx in range(2500):
        t = tickets[tid_idx]
        w = work_orders[tid_idx]
        
        t_open = datetime.strptime(t["dt_opened"], "%Y-%m-%dT%H:%M:%SZ")
        w_disp = datetime.strptime(w["dt_dispatched"], "%Y-%m-%dT%H:%M:%SZ")
        dt_link = max(t_open, w_disp) + timedelta(minutes=5)
        
        links.append({
            "link_id": link_id,
            "ticket_id": t["ticket_id"],
            "work_order_id": w["work_order_id"],
            "relationship_type": "PRIMARY_RESOLUTION",
            "dt_linked": dt_link.strftime("%Y-%m-%dT%H:%M:%SZ")
        })
        link_set.add((t["ticket_id"], w["work_order_id"]))
        link_id += 1

    # Pass 2: Remaining 700 links (3200 - 2500 = 700)
    # Create 1:M (one ticket has second dispatch) and M:N (one work order covers multiple tickets)
    extra_needed = 3200 - len(links)
    cursor = 0
    while len(links) < 3200:
        cursor += 1
        t_idx = (cursor * 37) % 2500
        # Connect to work order from the remaining pool (2500 to 2999) or shared backbone WOs
        w_idx = 2500 + (cursor % 500) if cursor % 2 == 0 else (cursor % 100)
        
        t_cand = tickets[t_idx]
        w_cand = work_orders[w_idx]
        
        pair = (t_cand["ticket_id"], w_cand["work_order_id"])
        if pair not in link_set:
            link_set.add(pair)
            t_open = datetime.strptime(t_cand["dt_opened"], "%Y-%m-%dT%H:%M:%SZ")
            w_disp = datetime.strptime(w_cand["dt_dispatched"], "%Y-%m-%dT%H:%M:%SZ")
            dt_link = max(t_open, w_disp) + timedelta(minutes=10)
            
            rel_type = "SECONDARY_DISPATCH" if cursor % 2 == 0 else "CORRELATED_INCIDENT"
            links.append({
                "link_id": link_id,
                "ticket_id": t_cand["ticket_id"],
                "work_order_id": w_cand["work_order_id"],
                "relationship_type": rel_type,
                "dt_linked": dt_link.strftime("%Y-%m-%dT%H:%M:%SZ")
            })
            link_id += 1
    print(f"  [+] Generated {len(links)} Ticket-Work-Order Bridge Links")

    # 6. Work Order Activities (6,000 rows)
    # Exactly 2 activity records per work order (3,000 * 2 = 6,000)
    activities = []
    act_id = 1
    for w in work_orders:
        w_disp = datetime.strptime(w["dt_dispatched"], "%Y-%m-%dT%H:%M:%SZ")
        
        # Activity 1: Dispatch / Arrival (15-30 mins after dispatch)
        act1_time = w_disp + timedelta(minutes=15)
        act1_type, act1_desc = ACTIVITY_SEQUENCES[1]
        activities.append({
            "activity_id": act_id,
            "work_order_id": w["work_order_id"],
            "activity_seq": 1,
            "activity_type": act1_type,
            "log_detail": act1_desc,
            "operator_alias": "Field_Engineer_Tech",
            "timestamp": act1_time.strftime("%Y-%m-%dT%H:%M:%SZ")
        })
        act_id += 1
        
        # Activity 2: Corrective Action / Closeout
        act2_time = act1_time + timedelta(minutes=45)
        act2_type, act2_desc = ACTIVITY_SEQUENCES[3] if w["status"] == "IN_PROGRESS" else ACTIVITY_SEQUENCES[5]
        activities.append({
            "activity_id": act_id,
            "work_order_id": w["work_order_id"],
            "activity_seq": 2,
            "activity_type": act2_type,
            "log_detail": f"{act2_desc} [{w['specific_cause'][:50]}]",
            "operator_alias": "NOC_Closeout_Desk",
            "timestamp": act2_time.strftime("%Y-%m-%dT%H:%M:%SZ")
        })
        act_id += 1
    print(f"  [+] Generated {len(activities)} Work Order Activities")

    # ── Export Datasets ───────────────────────────────────────────────────────
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    datasets = {
        "customers": customers,
        "circuits": circuits,
        "trouble_tickets": tickets,
        "work_orders": work_orders,
        "ticket_work_order_links": links,
        "work_order_activities": activities
    }
    
    for name, data in datasets.items():
        # Export CSV
        csv_path = os.path.join(OUTPUT_DIR, f"{name}.csv")
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            if data:
                writer = csv.DictWriter(f, fieldnames=list(data[0].keys()))
                writer.writeheader()
                writer.writerows(data)
                
        # Export JSON
        json_path = os.path.join(OUTPUT_DIR, f"{name}.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            
    print(f"[*] Successfully saved all 6 tables to {OUTPUT_DIR}")

    # ── Ground-Truth Fixtures ─────────────────────────────────────────────────
    fixtures = generate_ground_truth_fixtures(datasets)
    with open(FIXTURES_FILE, "w", encoding="utf-8") as f:
        json.dump(fixtures, f, ensure_ascii=False, indent=2)
    print(f"[*] Ground-truth fixtures saved to {FIXTURES_FILE}")
    
    return datasets


def generate_ground_truth_fixtures(datasets):
    """Generates a deterministic collection of verified known-answer test cases"""
    customers = {c["customer_id"]: c for c in datasets["customers"]}
    circuits = {c["circuit_id"]: c for c in datasets["circuits"]}
    tickets = {t["ticket_id"]: t for t in datasets["trouble_tickets"]}
    work_orders = {w["work_order_id"]: w for w in datasets["work_orders"]}
    
    # Map links
    t_to_w = {}
    w_to_t = {}
    for l in datasets["ticket_work_order_links"]:
        t_to_w.setdefault(l["ticket_id"], []).append(l["work_order_id"])
        w_to_t.setdefault(l["work_order_id"], []).append(l["ticket_id"])

    fixtures = {
        "metadata": {
            "description": "NOC Intelligence Ground-Truth Verification Fixtures",
            "seed": DEFAULT_SEED,
            "created_at": "2026-10-11T18:25:00Z"
        },
        "test_cases": [
            # 1. Direct Ticket Lookup
            {
                "case_id": "TC-01-DIRECT-TICKET",
                "category": "direct_ticket_lookup",
                "query": "Ticket INC-2025-00001 สถานะเป็นอะไร",
                "input_identifier": "INC-2025-00001",
                "expected": {
                    "ticket_id": "INC-2025-00001",
                    "customer_id": tickets["INC-2025-00001"]["customer_id"],
                    "customer_name": customers[tickets["INC-2025-00001"]["customer_id"]]["company_name_en"],
                    "status": tickets["INC-2025-00001"]["status"],
                    "severity": tickets["INC-2025-00001"]["severity"],
                    "circuit_ref": tickets["INC-2025-00001"]["circuit_ref"]
                }
            },
            # 2. Direct Work Order Lookup
            {
                "case_id": "TC-02-DIRECT-WO",
                "category": "direct_work_order_lookup",
                "query": "Work Order WO-700001 สาเหตุและอุปกรณ์ที่มีปัญหาคืออะไร",
                "input_identifier": "WO-700001",
                "expected": {
                    "work_order_id": "WO-700001",
                    "root_cause_category": work_orders["WO-700001"]["root_cause_category"],
                    "equipment_type": work_orders["WO-700001"]["equipment_type"],
                    "equipment_model": work_orders["WO-700001"]["equipment_model"],
                    "assigned_team": work_orders["WO-700001"]["assigned_team"],
                    "status": work_orders["WO-700001"]["status"]
                }
            },
            # 3. Bridge Traversal (Ticket to Work Order)
            {
                "case_id": "TC-03-BRIDGE-TICKET-TO-WO",
                "category": "bridge_traversal",
                "query": "ตั๋ว INC-2025-00042 เชื่อมโยงกับใบงานช่างใบไหนบ้าง",
                "input_identifier": "INC-2025-00042",
                "expected": {
                    "ticket_id": "INC-2025-00042",
                    "linked_work_order_ids": t_to_w.get("INC-2025-00042", []),
                    "work_order_count": len(t_to_w.get("INC-2025-00042", []))
                }
            },
            # 4. Bridge Traversal (Shared Backbone Work Order to Multiple Tickets)
            {
                "case_id": "TC-04-BRIDGE-SHARED-WO",
                "category": "bridge_traversal_multi",
                "query": "ใบงาน WO-700001 กระทบกับตั๋วลูกค้าใบไหนบ้าง",
                "input_identifier": "WO-700001",
                "expected": {
                    "work_order_id": "WO-700001",
                    "affected_ticket_ids": w_to_t.get("WO-700001", []),
                    "affected_ticket_count": len(w_to_t.get("WO-700001", []))
                }
            },
            # 5. Customer Ticket Counts
            {
                "case_id": "TC-05-CUSTOMER-COUNTS",
                "category": "customer_ticket_counts",
                "query": "Apex Logistics มีตั๋วแจ้งปัญหาทั้งหมดกี่รายการ",
                "input_company": "Apex Logistics",
                "expected": {
                    "customer_id": 1,
                    "company_name": "Apex Logistics",
                    "total_tickets": sum(1 for t in datasets["trouble_tickets"] if t["customer_id"] == 1),
                    "open_tickets": sum(1 for t in datasets["trouble_tickets"] if t["customer_id"] == 1 and t["status"] == "IN_PROGRESS"),
                    "resolved_tickets": sum(1 for t in datasets["trouble_tickets"] if t["customer_id"] == 1 and t["status"] in ["RESOLVED", "CLOSED"])
                }
            },
            # 6. Circuit to Customer Resolution
            {
                "case_id": "TC-06-CIRCUIT-RESOLUTION",
                "category": "circuit_to_customer",
                "query": "วงจร CKT-880001 เป็นของบริษัทอะไร และอยู่ที่สาขาไหน",
                "input_circuit_ref": "CKT-880001",
                "expected": {
                    "circuit_ref": "CKT-880001",
                    "customer_id": circuits[1]["customer_id"],
                    "company_name_en": customers[circuits[1]["customer_id"]]["company_name_en"],
                    "site_name": circuits[1]["site_name"],
                    "bandwidth_mbps": circuits[1]["bandwidth_mbps"]
                }
            },
            # 7. Missing Records (Negative Test)
            {
                "case_id": "TC-07-MISSING-TICKET",
                "category": "missing_records",
                "query": "Ticket INC-2025-99999 สถานะเป็นอะไร",
                "input_identifier": "INC-2025-99999",
                "expected": {
                    "found": False,
                    "expected_error": "NOT_FOUND"
                }
            },
            # 8. Ambiguous Natural Language Query
            {
                "case_id": "TC-08-AMBIGUOUS-QUERY",
                "category": "ambiguous_questions",
                "query": "เน็ตช้ามาก ใช้งานไม่ได้ ช่วยดูให้หน่อยครับ",
                "input_query": "เน็ตช้ามาก ใช้งานไม่ได้ ช่วยดูให้หน่อยครับ",
                "expected": {
                    "requires_clarification": True,
                    "clarification_type": "MISSING_COMPANY_OR_CIRCUIT"
                }
            }
        ]
    }
    return fixtures


if __name__ == "__main__":
    generate_all_datasets(DEFAULT_SEED)
