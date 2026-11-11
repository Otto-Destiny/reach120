# Reach120 build plan

Three weeks, one person, Kano first and Jigawa second. Each phase ends with a test you can run.

## Progress (11 November 2026)

| Phase | State | Evidence |
|---|---|---|
| 1. Set up and prepare one state | Done for Kano | `reach120 prepare`; windows, facilities and capable sets in `outputs/kano/inputs/` |
| 2. Score the capability stand-in | Done | `reach120 scorecard`: catch rates 6% / 15% / 63%; loose has the best balance and is the default |
| 3. Travel-time engine | Done | Compiled search with cap and exact pruning; tests pass, including the 30 km road and the SciPy comparison |
| 4. Validate | Done, except DHS | OnTIME: rank correlation 0.82-0.83 over 8,502 city cells, median 4.5-5.4 min faster than Google. OSRM: 0.82 at 50 rural origins |
| 5. Coverage tables | Done | Ward totals match the population layer exactly; state total within 0.1% of WorldPop's |
| 6. Upgrade optimizer | Done | Exact solver matches brute force in tests and never trails greedy |
| 7. Sensitivity and second state | Sensitivity done; Jigawa not yet run | `reach120 all --state Jigawa` needs about 1 GB of disk |
| 8. Briefs | Done | `outputs/kano/brief/reach120_kano_brief.pdf` |
| 9. App | Done | Five pages; each runs in the test suite and with the geospatial stack blocked |
| 10. Assistant | Built; not yet scored | Needs a free key to run the 30-question test (`reach120 ask`) |
| 11. Publish | Bundle ready | `reach120 bundle --state Kano`; deploying needs your Hugging Face account |

Changes from the plan below, found while building:

- The planner runs for both transport modes. In Kano a vehicle puts almost every woman within an
  hour, so upgrades only matter for women on foot.
- Travel time goes onto the population grid as a median, not a mean (see ARCHITECTURE.md).
- The least-cost search is a compiled implementation rather than scikit-image's, for the cap and
  the pruning.

## Before day 1

- [ ] Register with the DHS Program and request Nigeria 2023-24 with the GPS file. Approval takes
      days, so start now. The build does not wait for it.
- [ ] Get a free-tier language-model key for the assistant (Groq's free tier, or any
      OpenAI-compatible provider). Nothing else in the build needs it.
- [ ] Decide where upgrade cost figures come from (see Decisions).

## Week 1: travel times you can defend

### Phase 1. Set up and prepare one state (days 1-2)

- Create the repository, environment and `config/states.yaml`.
- Write the friction window reader: cut Kano plus a 50 km margin from the zipped surface, for both
  the mixed and the walking version.
- Load facilities v4.0 for the window. Clean types, drop Not-Functional and Closed.
- Build the strict, base and loose capable sets.

**Done when:** `reach120 prepare --state Kano` writes two friction windows and three capable-set
files, and a plot of each looks right against a base map.

### Phase 2. Score the capability stand-in (day 2)

- Match the OnTIME verified list to GRID3 facilities in the 15 cities by name similarity and
  distance (within 200 m, then by hand for the leftovers in Kano).
- For each scenario, report the share of verified facilities caught and the share of included
  facilities that are not on the verified list.

**Done when:** a one-table scorecard exists and the default scenario is chosen from it.

### Phase 3. Travel-time engine (days 3-4)

- Implement multi-source least-cost travel time.
- Unit tests on synthetic surfaces: a straight road, a river with one bridge, an unreachable island.
- Run Kano: 3 scenarios × 2 surfaces.

**Done when:** tests pass, including "30 km at 60 km/h is 30 minutes", and six travel-time
rasters exist for Kano.

### Phase 4. Validate (day 5)

- **Against Google travel times.** In the Kano conurbation, compare modelled time to the nearest
  verified facility with OnTIME's times for the same cells, under free-flow and peak traffic.
- **Against road routing.** For 50 rural origin-destination pairs, compare with a routing engine
  on OpenStreetMap roads.

**Done when:** `validation_report.md` gives rank correlation, median error in minutes, bias, and
the share of cells placed on the same side of 30 and 60 minutes. Write down the result even if it
is unflattering.

## Week 2: from map to decision

### Phase 5. Coverage tables (day 6)

- Put travel time on the population grid, count women within 30, 60 and 120 minutes by ward, LGA
  and state.

**Done when:** state totals match the sum of the population layer to within 0.1%, and the
headline "X% of women live more than 2 hours away" exists for each scenario.

### Phase 6. Upgrade optimizer (days 7-9)

- Build the candidate table and the cost table.
- Compute reach for each candidate (travel-time runs stopped at 120 minutes).
- Implement the exact covering model and the greedy ranking.
- Sweep budgets to produce the women-reached-per-naira curve.

**Done when:** on a toy case small enough to solve by hand, the optimizer returns the known best
answer; and on Kano the exact answer is never worse than greedy.

### Phase 7. Sensitivity and second state (day 10)

- Rerun the plan under each capability scenario and both friction surfaces.
- Run the whole pipeline for Jigawa with one command.

**Done when:** each headline number has a range, and Jigawa needed no code changes.

## Week 3: something people can use

### Phase 8. Briefs (day 11)

- Template the one-page brief: headline, map, top upgrades, budget curve, caveats.

**Done when:** `reach120 brief --state Kano` writes a PDF whose numbers match the tables exactly.

### Phase 9. App (days 12-13)

- Streamlit pages: Map, Planner, Ask, Method and limits.

**Done when:** moving the budget slider returns a new plan within 5 seconds.

### Phase 10. Assistant (day 14)

- Tool-calling loop over the engine, with the number check.
- Test set of 30 questions with known answers, including 5 it should decline (clinical advice,
  states not loaded).
- Run the test set on two or three free-tier models and keep the best.

**Done when:** at least 27 of 30 are right, none contains a number absent from a tool result, and
all 5 out-of-scope questions are declined.

### Phase 11. Publish (day 15)

- Deploy to Hugging Face Spaces.
- Write-up with the validation results and limits.
- How-to guide: "Run Reach120 for your state".

**Done when:** the three links work from a phone.

## What gets reported

| Claim | Evidence |
|---|---|
| Travel times are realistic | Correlation and median error against Google times in Kano; routing spot-check on rural pairs |
| The capability stand-in is reasonable | Catch rate and false-inclusion rate against the verified list in 15 cities |
| The answer is robust | Range of the headline across 3 scenarios × 2 surfaces |
| The optimizer adds value | Women reached versus greedy and versus "upgrade the biggest hospitals first" |
| It predicts behaviour (if DHS arrives) | Home-birth share by modelled travel-time band in DHS clusters |

## Risks

| Risk | Response |
|---|---|
| Matching OnTIME names to GRID3 is messy | Limit hand-matching to Kano; report the automatic match rate for the other cities |
| The model is much faster than Google inside the city | Report it. Rural access is the target; say the model is a dry-season, no-traffic best case |
| Cost figures are weak | Keep them editable and show results as "per ₦100m" as well as totals |
| DHS approval is slow | The DHS check is a bonus, not a gate |
| Memory runs short on big states | Tile the state with overlap; Kano and Jigawa fit as they are |

## Decisions for you

1. **Pilot states.** Default: Kano, then Jigawa.
2. **Cost source.** Default: three upgrade tiers with placeholder costs you replace from a state
   budget or a published costing study.
3. **Default capability scenario.** Chosen in phase 2 from the scorecard, not before.

## Not in these three weeks

The voice line in Hausa, Igbo or Yoruba using N-ATLaS. It reuses the travel-time engine, so it is
a natural second release once the planner is live.
