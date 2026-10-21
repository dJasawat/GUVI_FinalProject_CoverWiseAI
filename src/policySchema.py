from dataclasses import asdict, dataclass

@dataclass(frozen=True)
class PolicyChunk:
    chunk_id: str
    policy_name: str
    page_number: int
    page_numbers: list[int]
    clause_heading: str
    section_title: str
    text: str

@dataclass
class PolicyFeatures:
    policy_id: str
    policy_name: str

    sum_insured: float | None
    annual_premium: float | None

    waiting_period_months: int | None
    ped_waiting_period_months: int | None

    copayment_percentage: float | None

    room_rent_limit_present: bool
    icu_limit_present: bool

    disease_sub_limit_count: int
    exclusion_count: int

    restoration_benefit_present: bool
    no_claim_bonus_present: bool

    child_coverage_present: bool
    maternity_coverage_present: bool

    claim_process_clarity_score: float
    ambiguous_clause_count: int
    risk_flags_count: int