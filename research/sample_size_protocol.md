# Risk-score sample-size validation protocol

## Claim boundary

The automated pilot measures **sampling stability** against each product's full
captured review pool. It does not establish classifier accuracy, population
representativeness, or validity of the 30/70 risk formula.

The validation population is deliberately limited to **electronics and gadgets
sold on Lazada Philippines**. Findings must not be generalized to other product
categories or marketplaces without a separate external validation.

This domain matches the current risk lexicon particularly well: charging and
battery failures, connectivity and performance faults, overheating and other
safety issues, breakage and durability, material quality, and counterfeit
claims. VADER alone is not the basis for this scope; the deployed system also
uses the FiReCS classifier and Filipino/Taglish lexicons.

## Product strata

Use five strata so one gadget type cannot dominate the result:

- Audio: earphones, headphones, speakers, microphones.
- Power: chargers, cables, power banks, batteries, adapters.
- Mobile accessories: cases, mounts, screen protectors, mobile peripherals.
- Computer peripherals: keyboards, mice, webcams, hubs, storage devices.
- Smart and home gadgets: cameras, smart plugs, lights, small connected devices.

The first validation pass should contain at least six eligible products per
stratum (30 total). Prefer 10 per stratum (50 total) when collection and human
annotation capacity allow it.

## Required final dataset

- At least 30 products for the first validation pass; target 50–100.
- Lazada Philippines only, distributed across the five electronics-and-gadgets
  strata above.
- For production scoring, compile every successfully collected, non-empty,
  deduplicated written review when 100 or fewer are available. When more than
  100 are available, compile a representative maximum of 100 reviews per
  product.
- Do not prioritize low-star or 3-star reviews for production scoring. A
  low-star-first queue may be used for model-error research, but it must remain
  separate from the reviews used to calculate a product score.
- For products with more than 100 reviews, sample proportionally from the
  marketplace's actual 1–5-star distribution. Record the displayed star counts,
  sampled counts, total available reviews, usable reviews, and exclusions. If a
  stratum is deliberately oversampled, restore representativeness with the
  documented weight `displayed count / sampled count` for that star level.
- Sample across available review pages or time periods within each star level so
  the first page does not dominate the result.
- Two independent human annotators plus adjudication for disagreements.
- Fields: product ID, marketplace, review ID, review text, rating, sentiment
  label, risk detected, severity, negation, duplicate, annotators, sampling
  stratum, sampling weight, and collection timestamp. Risk category may be kept
  for research but is not required by the production risk score.

## Production sampling and reporting rule

For each product, the number of compiled reviews is:

```text
min(all usable written reviews available, 100)
```

The app must report coverage as `reviews analyzed / reviews available`, for
example `73 / 73` or `100 / 2,000`. It must also report a bootstrap 95% confidence
interval for the risk score. Mark results based on fewer than 30 usable reviews,
or samples with an unknown or demonstrably biased star distribution, as limited
evidence. The final confidence wording and thresholds must be calibrated in the
sample-size study rather than treated as model accuracy.

The 100-review cap applies to a product analysis request. It does not cap the
number of reviews retained across products for model training. Offline sampling
research may retain larger full-product pools as references, but those pools do
not change the production collection limit.

## Frozen acceptance rule

Test sample sizes 10, 20, 30, 50, 75, and 100 with 1,000 repeated,
star-representative samples per product. The provisional rule selects the
smallest size where:

1. the 90th-percentile absolute score error is no more than 5 points; and
2. Low/Moderate/High agreement with the gold reference is at least 90%.

Report results overall and by gadget stratum and distance from the 30/60
decision boundaries. Confirm the selected size on unseen Lazada electronics
products.

## Run the pilot

```powershell
.\.venv\Scripts\python.exe .\research\sample_size_study.py --json <files...>
```

Use `--annotations annotations.csv` once adjudicated human labels are ready.
