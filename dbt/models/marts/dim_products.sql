select product_id, brand, therapeutic_area, cast(is_own_brand as boolean) as is_own_brand,
       cast(list_price as decimal(10, 2)) as list_price
from {{ ref('raw_products') }}
