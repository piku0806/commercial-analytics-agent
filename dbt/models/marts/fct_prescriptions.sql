select rx_id, cast(rx_date as date) as rx_date, prescriber_id, product_id, payer_type,
       cast(quantity as integer) as quantity, cast(net_sales as decimal(12, 2)) as net_sales,
       cast(is_new_rx as integer) as is_new_rx
from {{ ref('raw_prescriptions') }}
