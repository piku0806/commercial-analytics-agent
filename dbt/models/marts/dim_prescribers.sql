select prescriber_id, prescriber_name, specialty, territory, region from {{ ref('raw_prescribers') }}
