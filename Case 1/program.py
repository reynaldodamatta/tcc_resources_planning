import json
import os
import random
import sys
from datetime import date, timedelta

try:
    from deap import algorithms, base, creator, tools
except ImportError:
    with open("OptimizationMessage.json", "w", encoding="utf-8") as _f:
        json.dump({"success": False, "message": "DEAP not found. Install with: pip install deap"}, _f)
    sys.exit(1)

# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------

DAILY_LABOR_HOURS = 8

INPUT_FILE  = "InputData.json"
OUTPUT_FILE = "OutputData.json"
ERROR_FILE  = "OptimizationMessage.json"

# ---------------------------------------------------------------------------
# GA hyper-parameters
# ---------------------------------------------------------------------------

RANDOM_SEED        = 42     # fixed seed to guarantee reproducibility 
POPULATION_SIZE    = 80
MAX_GENERATIONS    = 150
CROSSOVER_PROB     = 0.70
MUTATION_PROB      = 0.30
MUTATION_GENE_PROB = 0.30   # probability of mutating each individual gene
TOURNAMENT_SIZE    = 3
HALL_OF_FAME_SIZE  = 3      # The first three chromosomes per activity are kept

# Penalty factors (applied to base slot cost)
OUTSOURCE_MULTIPLIER  = 3.0  # Cost for outsourcing. Triples the cost (cost x 3) when no employee is available
POSITION_PENALTY_STEP = 0.5  # added per seniority level below required
                              # e.g. 1 level below → ×1.5 | 2 levels → ×2.0 | 3 → ×2.5
                              # Candidates above the required level are NOT penalised because
                              # their higher cost already makes them less attractive.

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)


# ---------------------------------------------------------------------------
# Date helpers
# ---------------------------------------------------------------------------

def add_working_days(start: date, days: int) -> date:
    """Return the date that is exactly `days` working days after `start`."""
    if days == 0:
        while start.weekday() >= 5:         
            start += timedelta(days=1)
        return start
    added, current = 0, start
    while added < days:
        current += timedelta(days=1)
        if current.weekday() < 5:
            added += 1
    return current


# ---------------------------------------------------------------------------
# Input loading
# ---------------------------------------------------------------------------

def load_input(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def build_position_daily_cost_lookup(position_order: list) -> dict:
    """Returns position → daily cost.
    Example: {"Analyst": 12.0, "Specialist": 32.0, ...}
    """
    return {p["name"]: float(p["hourly_cost"]) * DAILY_LABOR_HOURS for p in position_order}

def build_position_order_lookup(position_order: list) -> dict:
    """Returns name → seniority order (integer).
    Lower order = less senior, higher order = more senior.
    Example: {"Intern": 1, "Analyst": 2, "Lead Analyst": 3, ...}
    """
    return {p["name"]: int(p["order"]) for p in position_order}

# ---------------------------------------------------------------------------
# Candidate resolution per slot
# ---------------------------------------------------------------------------

def find_eligible_employees(expected_team_slot: dict, candidate_employees: list, position_order: dict) -> list:
    """
    Filter candidate_employees for a given expected_team slot.
    A candidate is valid when:
      1. They have at least 1 macro skill required for the slot.
      2. They have at least the number of free days required for the slot (estimated_person_days) .

    Candidates at ANY seniority level are accepted; the fitness function
    penalises them instead of excluding them:
      - Position: 
         - over-qualified (higher order): no extra penalty, their higher
           cost already makes them less attractive.
         - Under-qualified (lower order): penalty of 1.0 + (position_gap × POSITION_PENALTY_STEP).
      - Macro skills: penalty = required_count / matching_count
        (e.g. 3 required, 2 matched: ×1.5; all matched: ×1.0).

    Returns a list of (candidate_employee_slot, skill_match_score, skill_penalty, position_penalty).
    """
    required_skills  = {s.lower() for s in expected_team_slot["macro_skills"]}
    required_days    = expected_team_slot["estimated_person_days"]
    required_order   = position_order.get(expected_team_slot["position"], 0)

    eligible_candidates = []
    for candidate_employee_slot in candidate_employees:
        candidate_skills = {s.lower() for s in candidate_employee_slot["macro_skills"]}

        # Criterion 1 — at least one matching macro skill
        matching = required_skills & candidate_skills
        if not matching:
            continue

        # Criterion 2 — enough free days
        if len(candidate_employee_slot["free_days"]) < required_days:
            continue

        # Skill penalty: required / matching
        # (e.g. 3 required, 2 matched → ×1.5; fully matched → ×1.0)
        skill_score   = len(matching) / len(required_skills) if required_skills else 1.0
        skill_penalty = len(required_skills) / len(matching)

        # Position penalty
        # Over-qualified or exact match → 1.0 
        # Under-qualified → increases with each seniority level below the requirement
        candidate_order  = position_order.get(candidate_employee_slot["position"], 0)
        
        position_gap = required_order - candidate_order
        if position_gap > 0:
            position_penalty = 1.0 + (position_gap * POSITION_PENALTY_STEP)
        else:
            position_penalty = 1.0

        eligible_candidates.append((candidate_employee_slot, skill_score, skill_penalty, position_penalty))

    return eligible_candidates

# ---------------------------------------------------------------------------
# Fitness evaluation
# ---------------------------------------------------------------------------

def compute_cost(individual: list, expected_team_slots: list, eligible_candidates_per_slot: list, position_cost: dict) -> tuple:
    """
    Compute the total weighted cost for one chromosome.
    
    Cost per slot:
      - Assigned employee → candidate_cost x days x skill_penalty x position_penalty
          * candidate_cost: actual candidate cost (higher for over-qualified — natural penalty)
          * skill_penalty: required_skills / matching_skills (≥ 1.0)
          * position_penalty: 1.0 if over/equal, 1.0 + position_gap x STEP if under-qualified
      - No employee (outsource) → base_cost x days x OUTSOURCE_MULTIPLIER
    """
    total = 0.0
    for i, gene in enumerate(individual):
        expected_team_slot = expected_team_slots[i]
        eligible_candidates = eligible_candidates_per_slot[i]
        days               = expected_team_slot["estimated_person_days"]

        # Required position's cost — used as reference for outsourcing cost
        base_cost = position_cost.get(expected_team_slot["position"])
        if base_cost is None:
            raise ValueError(f"Position '{expected_team_slot['position']}' not found in cost table.")

        if gene >= len(eligible_candidates):          # no candidate available → outsource
            total += base_cost * days * OUTSOURCE_MULTIPLIER
        else:
            candidate_employee_slot, skill_score, skill_penalty, position_penalty = eligible_candidates[gene]
            candidate_cost = position_cost.get(candidate_employee_slot["position"], base_cost)

            # For the GA's fitness signal we use max(candidate_cost, base_cost) so that
            # under-qualified candidates (cheaper daily rate) are still charged at least
            # the slot's expected rate before the position penalty is applied.
            # This ensures an under-qualified candidate always costs MORE (in penalised
            # terms) than the ideal candidate, regardless of how cheap they actually are:
            #
            #   Under-qualified: max(candidate_cost, base_cost) × days × skill_p × pos_p
            #                  = base_cost × days × (>1.0) × (>1.0)  → more expensive
            #   Exact match:     max(base_cost, base_cost) × days × 1.0 × 1.0
            #                  = base_cost × days               → reference cost
            #   Over-qualified:  max(candidate_cost, base_cost) × days × ... × 1.0
            #                  = candidate_cost × days × ...    → higher cost is the penalty
            effective_cost = max(candidate_cost, base_cost)
            cost = effective_cost * days * skill_penalty * position_penalty
            total += cost

    # Tiebreaker: fewer outsourced slots wins.
    slots_outsourced = sum(
        1
        for slot_index, gene in enumerate(individual)
        if gene >= len(eligible_candidates_per_slot[slot_index])
    )
    OUTSOURCE_TIEBREAK_DELTA = 0.001   # small enough to never override the primary cost signal
    return (total + slots_outsourced * OUTSOURCE_TIEBREAK_DELTA,)


# ---------------------------------------------------------------------------
# Mutation operator
# ---------------------------------------------------------------------------

def mutate_genes(individual: list, n_options_per_slot: list, indpb: float = MUTATION_GENE_PROB) -> tuple:
    """Per-gene mutation: randomly reassign a slot candidate."""
    for i in range(len(individual)):
        if random.random() < indpb:
            individual[i] = random.randrange(n_options_per_slot[i])
    return (individual,)  # DEAP requires mutation functions to return a tuple


# ---------------------------------------------------------------------------
# Duplicate-employee repair
# ---------------------------------------------------------------------------

def repair_duplicates(individual: list, eligible_candidates_per_slot: list) -> None:
    """
    Ensure no employee is assigned to more than one slot in the same activity.
    When a conflict is found the later slot is reassigned to another available
    candidate; if none is free it falls back to outsource (last gene index).
    Modifies `individual` in-place.
    """
    used_ids = set()
    for i in range(len(individual)):
        gene                = individual[i]
        eligible_candidates = eligible_candidates_per_slot[i]

        if gene >= len(eligible_candidates):
            continue  # already outsourced — no conflict possible

        candidate_employee_slot, *_ = eligible_candidates[gene]
        emp_id                      = candidate_employee_slot["id"]

        if emp_id in used_ids:
            # Try to find another candidate for this slot that is not yet assigned
            replaced = False
            for alt_gene, (alt_candidate_employee_slot, *_) in enumerate(eligible_candidates):
                if alt_candidate_employee_slot["id"] not in used_ids:
                    individual[i] = alt_gene
                    used_ids.add(alt_candidate_employee_slot["id"])
                    replaced = True
                    break
            if not replaced:
                individual[i] = len(eligible_candidates)  # outsource — no unique candidate available
        else:
            used_ids.add(emp_id)


# ---------------------------------------------------------------------------
# Per-activity GA
# ---------------------------------------------------------------------------

def optimize_activity(expected_team_slots: list, eligible_candidates_per_slot: list, position_cost: dict) -> tuple:
    """
    Run the GA for one activity.
    Returns (best_individual, best_fitness, hall_of_fame).
    The HallOfFame retains the top-3 chromosomes for the final report.
    """
    if not expected_team_slots:
        return [], 0.0, tools.HallOfFame(HALL_OF_FAME_SIZE)

    # +1 per slot to include the OUTSOURCE option (last index)
    n_options = [len(c) + 1 for c in eligible_candidates_per_slot]

    def create_individual():
        ind = creator.Individual([random.randrange(n) for n in n_options])
        repair_duplicates(ind, eligible_candidates_per_slot)
        return ind

    def mate_and_repair(ind1, ind2):
        tools.cxUniform(ind1, ind2, indpb=0.5)
        repair_duplicates(ind1, eligible_candidates_per_slot)
        repair_duplicates(ind2, eligible_candidates_per_slot)
        return ind1, ind2

    def mutate_and_repair(ind):
        mutate_genes(ind, n_options)
        repair_duplicates(ind, eligible_candidates_per_slot)
        return (ind,)

    toolbox = base.Toolbox()
    toolbox.register("individual", create_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register(
        "evaluate", compute_cost,
        expected_team_slots=expected_team_slots,
        eligible_candidates_per_slot=eligible_candidates_per_slot,
        position_cost=position_cost
    )
    toolbox.register("mate",   mate_and_repair)
    toolbox.register("mutate", mutate_and_repair)
    toolbox.register("select", tools.selTournament, tournsize=TOURNAMENT_SIZE)

    random.seed(RANDOM_SEED)
    pop = toolbox.population(n=POPULATION_SIZE)
    hof = tools.HallOfFame(HALL_OF_FAME_SIZE)

    algorithms.eaSimple(
        pop, toolbox,
        cxpb=CROSSOVER_PROB,
        mutpb=MUTATION_PROB,
        ngen=MAX_GENERATIONS,
        halloffame=hof,
        verbose=False
    )

    best = hof[0]
    return best, float(best.fitness.values[0]), hof


# ---------------------------------------------------------------------------
# Solution builder
# ---------------------------------------------------------------------------

def build_solution(individual: list, expected_team_slots: list, eligible_candidates_per_slot: list,
                   position_cost: dict) -> tuple:
    """
    Translate one GA solution into lists of assignment and position_gap dicts.
    Also computes the total real cost (no penalties) so it can be reported
    alongside the penalised fitness cost used by the GA.

    slot_index  — 0-based index of the slot within this activity's expected_team list.
    real_cost per slot:
      - Assignment:   actual_price x allocated_days  (no skill/position penalties)
      - position_gap: base_cost x required_days x OUTSOURCE_MULTIPLIER
    """
    assignments = []
    gaps        = []
    real_cost   = 0.0

    for i, gene in enumerate(individual):
        expected_team_slot  = expected_team_slots[i]
        eligible_candidates = eligible_candidates_per_slot[i]
        slot_index          = i               # 0-based index within this activity's slot list
        required_days       = expected_team_slot["estimated_person_days"]
        base_cost          = position_cost.get(expected_team_slot["position"], 0.0)

        if gene >= len(eligible_candidates):              # outsourced / position_gap
            real_cost += base_cost * required_days * OUTSOURCE_MULTIPLIER
            gaps.append({
                "slot_index":            slot_index,
                "role":                  expected_team_slot["role"],
                "required_position":     expected_team_slot["position"],
                "required_macro_skills": expected_team_slot["macro_skills"],
            })
        else:
            candidate_employee_slot, skill_score, *_ = eligible_candidates[gene]
            candidate_cost  = position_cost.get(candidate_employee_slot["position"], base_cost)
            real_cost       += candidate_cost * required_days
            required_skills_set  = {s.lower() for s in expected_team_slot["macro_skills"]}
            candidate_skills_set = {s.lower() for s in candidate_employee_slot["macro_skills"]}
            matched_skills       = sorted(required_skills_set & candidate_skills_set)
            assignments.append({
                "slot_index":            slot_index,
                "role":                  expected_team_slot["role"],
                "position":              expected_team_slot["position"],
                "employee_id":           candidate_employee_slot["id"],
                "employee_name":         candidate_employee_slot["name"],
                "employee_position":     candidate_employee_slot["position"],
                "skill_match_score":     round(skill_score, 4),
                "required_macro_skills": expected_team_slot["macro_skills"],
                "matched_macro_skills":  matched_skills,
                "allocated_dates":       candidate_employee_slot["free_days_date_format"][:required_days],
            })

    return assignments, gaps, real_cost


# ---------------------------------------------------------------------------
# Main optimisation loop
# ---------------------------------------------------------------------------

def optimize(data: dict) -> dict:
    proposal       = data["proposal"]
    position_order = data["position_order"]

    # Build lookup tables 
    # position_cost:  daily rate   e.g. {"Analyst": 96.0, "Specialist": 256.0}
    # position_order: seniority    e.g. {"Intern": 1, "Analyst": 2, ...}
    position_cost  = build_position_daily_cost_lookup(position_order)
    position_order = build_position_order_lookup(position_order)

    # Validate project start date early so errors surface cleanly
    project_start = date.fromisoformat(proposal["project_start_date"])

    deliverables = sorted(proposal["deliverables"], key=lambda d: d["deliverable_order"])

    # Total deadline expressed as calendar days (weekends included)
    total_deadline_wd = sum(d["deadline_in_days"] for d in deliverables)
    deadline_days     = (add_working_days(project_start, total_deadline_wd) - project_start).days

    deliverables_output = []
    total_slot_count    = 0          # running count of all slots across all activities
    total_fitness       = 0.0
    max_end_day         = 0          # calendar-day offset of the latest activity end
    outsourced_count    = 0
    cumulative_wd       = 0          # internal working-day offset from project start

    for deliverable in deliverables:
        deliverable_start_wd = cumulative_wd
        act_cum_wd           = 0
        activities_output    = []

        for activity in deliverable["activities"]:
            # Internal working-day offsets — used only to drive the GA window logic
            act_start_wd = deliverable_start_wd + act_cum_wd
            act_end_wd   = act_start_wd + activity["estimated_duration_days"]

            # Calendar-day offsets for the output — weekends are now counted so that
            # project_start + start_day (timedelta) lands on the real calendar date.
            act_start_day = (add_working_days(project_start, act_start_wd) - project_start).days
            act_end_day   = (add_working_days(project_start, act_end_wd)   - project_start).days

            expected_team_slots  = activity["expected_team"]
            candidate_employees  = activity["candidate_employees"]

            eligible_candidates_per_slot = [
                find_eligible_employees(expected_team_slot, candidate_employees, position_order)
                for expected_team_slot in expected_team_slots
            ]

            best, fitness, hof = optimize_activity(expected_team_slots, eligible_candidates_per_slot, position_cost)

            solutions = []
            for rank, solution in enumerate(hof, start=1):
                penalized_cost = float(solution.fitness.values[0])
                assignments, gaps, real_cost = build_solution(
                    solution, expected_team_slots, eligible_candidates_per_slot, position_cost
                )
                solutions.append({
                    "rank":        rank,
                    "cost":        round(penalized_cost, 4),  # GA fitness (includes penalties)
                    "real_cost":   round(real_cost, 4),       # actual cost without penalties
                    "assignments": assignments,
                    "gaps":        gaps,
                })

            # activity status
            gap_count   = len(solutions[0]["gaps"]) if solutions else len(expected_team_slots)
            status_text = (
                "✓ Fully staffed"
                if gap_count == 0
                else f"⚠ 1 role needs outsourcing"
                if gap_count == 1
                else f"⚠ {gap_count} roles need outsourcing"
            )

            activities_output.append({
                "name":      activity["name"],
                "start_day": act_start_day,
                "end_day":   act_end_day,
                "status":    status_text,
                "solutions": solutions,
            })

            # Counters are based on the best solution (rank 1)
            total_fitness    += fitness
            max_end_day       = max(max_end_day, act_end_day)
            outsourced_count += gap_count
            total_slot_count += len(expected_team_slots)
            act_cum_wd       += activity["estimated_duration_days"]

        # Per-deliverable real cost (sum of best-solution real costs)
        deliverable_cost = sum(
            act["solutions"][0]["real_cost"]
            for act in activities_output
            if act["solutions"]
        )

        deliverables_output.append({
            "order":            deliverable["deliverable_order"],
            "name":             deliverable["deliverable_name"],
            "deadline_in_days": deliverable["deadline_in_days"],
            "total_cost_usd":   round(deliverable_cost, 2),
            "activities":       activities_output,
        })

        cumulative_wd += act_cum_wd

    viable          = outsourced_count == 0
    within_deadline = max_end_day <= deadline_days

    # Ideal cost = every slot filled at base cost with a perfect match (no penalties).
    # This is the minimum the GA can achieve.
    ideal_total_cost = sum(
        position_cost.get(expected_team_slot["position"], 0.0) * expected_team_slot["estimated_person_days"]
        for deliverable in deliverables
        for activity in deliverable["activities"]
        for expected_team_slot in activity["expected_team"]
    )

    # fitness_score = ideal / penalised  →  1.0 means every slot was filled with an
    # exact position+skill match and no outsourcing; values drop toward 0 as
    # penalties (skill mismatch, position mismatch, outsourcing) increase.
    fitness_score = ideal_total_cost / total_fitness

    total_real_cost  = sum(d["total_cost_usd"] for d in deliverables_output)
    total_staffed    = total_slot_count - outsourced_count
    summary_status   = (
        "✓ Viable – all roles staffed"
        if viable
        else f"⚠ Needs outsourcing – {outsourced_count} role(s) unfilled"
    )

    return {
        "viable":           viable,
        "fitness_score":    fitness_score,
        "generations_run":  MAX_GENERATIONS,
        "max_end_day":      max_end_day,
        "deadline_days":    deadline_days,
        "within_deadline":  within_deadline,
        "total_activities": total_slot_count,
        "outsourced_count": outsourced_count,
        "summary": {
            "status":                summary_status,
            "total_cost_usd":        round(total_real_cost, 2),
            "roles_staffed":         total_staffed,
            "roles_outsourced":      outsourced_count,
            "project_duration_days": max_end_day,
            "deadline_days":         deadline_days,
            "within_deadline":       within_deadline,
        },
        "deliverables":     deliverables_output,
    }


def write_error(message: str) -> None:
    with open(ERROR_FILE, "w", encoding="utf-8") as f:
        json.dump({"success": False, "message": message}, f, indent=4)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
   
    INPUT_FILE = "D:/Work/TCC/InputData.json"
    OUTPUT_FILE = "D:/Work/TCC/OutputData.json"

    try:
        # --- Colab: upload manual se o arquivo não existir ---
        if not os.path.exists(INPUT_FILE):
            try:
                from google.colab import files
                import shutil
                print("InputData.json não encontrado. Selecione o arquivo:")
                uploaded = files.upload()
                shutil.copy(list(uploaded.keys())[0], INPUT_FILE)
            except ImportError:
                # Não está no Colab — comportamento original
                write_error("InputData.json not found.")
                sys.exit(1)

        data   = load_input(INPUT_FILE)
        result = optimize(data)

        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=4, ensure_ascii=False)

        sys.exit(0)

    except Exception as exc:
        write_error(str(exc))
        sys.exit(1)


if __name__ == "__main__":
    main()
