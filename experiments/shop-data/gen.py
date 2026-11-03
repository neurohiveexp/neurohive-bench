import random, json, csv, io, os, datetime as dt
os.makedirs(os.path.join(os.path.dirname(__file__),"data"),exist_ok=True)
random.seed(42)
cities=["Москва","Санкт-Петербург","Новосибирск","Екатеринбург","Казань","Нижний Новгород","Самара","Краснодар","Воронеж","Пермь"]
cats=["Смартфоны","Наушники","Ноутбуки","Бытовая техника","Аксессуары","Умный дом","Телевизоры","Планшеты"]
pay=["card","sbp","cash_on_delivery","installment"]; st=["delivered","shipped","cancelled","returned","paid"]
t0=dt.datetime(2026,10,1)
rows=[]
for i in range(1000):
    t=t0+dt.timedelta(seconds=random.randint(0,30*86400))
    rows.append(dict(order_id=f"ORD-{2026100000+i}",created_at=t.strftime("%Y-%m-%d %H:%M:%S"),user_id=random.randint(100000,999999),
        city=random.choice(cities),category=random.choice(cats),sku=f"SKU-{random.randint(10000,99999)}",qty=random.randint(1,4),
        price_rub=round(random.uniform(490,189990),2),discount_pct=random.choice([0,0,0,5,10,15]),payment=random.choice(pay),status=random.choice(st)))
f=io.StringIO(); w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
open(os.path.join(os.path.dirname(__file__),"data","orders_1000.csv"),"w").write(f.getvalue())
md="| "+" | ".join(rows[0].keys())+" |\n|"+"---|"*len(rows[0])+"\n"+"".join("| "+" | ".join(str(v) for v in r.values())+" |\n" for r in rows)
open(os.path.join(os.path.dirname(__file__),"data","orders_1000.md"),"w").write(md)
open(os.path.join(os.path.dirname(__file__),"data","orders_1000.json"),"w").write(json.dumps(rows,ensure_ascii=False))
# Kafka-like clickstream events
evn=["page_view","view_item","add_to_cart","begin_checkout","purchase","search"]
dev=["mobile","desktop","tablet"]; src=["yandex","google","direct","telegram","vk","email"]
ev=[]
for i in range(1000):
    t=t0+dt.timedelta(milliseconds=random.randint(0,30*86400*1000))
    e=dict(event_id=f"{random.getrandbits(64):016x}",event_name=random.choice(evn),event_ts=t.isoformat(timespec="milliseconds")+"Z",
        user_id=str(random.randint(100000,999999)),session_id=f"s{random.getrandbits(40):010x}",
        page=dict(path=random.choice(["/","/catalog/smartfony","/product/12345","/cart","/checkout","/search"]),referrer=random.choice(src)),
        device=dict(type=random.choice(dev),os=random.choice(["Android 16","iOS 26","Windows 11","macOS 16"]),app_version="5.12.3"),
        geo=dict(country="RU",city=random.choice(cities)),
        item=dict(item_id=f"SKU-{random.randint(10000,99999)}",item_category=random.choice(cats),price=round(random.uniform(490,189990),2),quantity=random.randint(1,3)))
    ev.append(e)
open(os.path.join(os.path.dirname(__file__),"data","events_1000.jsonl"),"w").write("\n".join(json.dumps(e,ensure_ascii=False) for e in ev))
# BigQuery schema DDL
ddl='''CREATE TABLE shop.orders (
  order_id STRING NOT NULL OPTIONS(description="Номер заказа"),
  created_at TIMESTAMP NOT NULL OPTIONS(description="Время оформления, UTC"),
  user_id INT64 OPTIONS(description="ID покупателя"),
  city STRING OPTIONS(description="Город доставки"),
  payment STRING OPTIONS(description="Способ оплаты: card, sbp, cash_on_delivery, installment"),
  status STRING OPTIONS(description="Статус: paid, shipped, delivered, cancelled, returned"),
  total_rub NUMERIC OPTIONS(description="Сумма заказа после скидки, руб.")
) PARTITION BY DATE(created_at);
CREATE TABLE shop.order_items (
  order_id STRING NOT NULL, sku STRING NOT NULL OPTIONS(description="Артикул"),
  category STRING OPTIONS(description="Категория каталога"), qty INT64, price_rub NUMERIC, discount_pct INT64
);
CREATE TABLE shop.events (
  event_id STRING, event_name STRING OPTIONS(description="page_view, view_item, add_to_cart, begin_checkout, purchase, search"),
  event_ts TIMESTAMP, user_id STRING, session_id STRING,
  page STRUCT<path STRING, referrer STRING>,
  device STRUCT<type STRING, os STRING, app_version STRING>,
  geo STRUCT<country STRING, city STRING>,
  item STRUCT<item_id STRING, item_category STRING, price NUMERIC, quantity INT64>
) PARTITION BY DATE(event_ts) CLUSTER BY event_name OPTIONS(description="Поток событий из Kafka, топик shop.clickstream");
CREATE TABLE shop.users (
  user_id INT64, registered_at TIMESTAMP, city STRING, loyalty_tier STRING OPTIONS(description="none, silver, gold"),
  marketing_opt_in BOOL
);'''
open(os.path.join(os.path.dirname(__file__),"data","schema.sql"),"w").write(ddl)
# aggregated SQL result: 200 rows (day x category funnel)
res=["day,category,views,add_to_cart,purchases,revenue_rub,conversion_pct"]
for d in range(25):
    for c in cats:
        v=random.randint(800,20000); a=int(v*random.uniform(0.03,0.12)); p=int(a*random.uniform(0.2,0.5))
        res.append(f"2026-10-{d+1:02d},{c},{v},{a},{p},{round(p*random.uniform(5000,60000),2)},{round(p/v*100,2)}")
open(os.path.join(os.path.dirname(__file__),"data","sql_result_200.csv"),"w").write("\n".join(res))
