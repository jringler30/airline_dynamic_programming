# ==============================================================
# MSBA – Optimization 2: Project 2
# Airline Ticket Pricing via Dynamic Programming
# ==============================================================
#
# OVERVIEW:
#   We model 365 days of ticket sales on a single flight.
#   The airline chooses a price each day for coach and for
#   first-class tickets. Demand is probabilistic — the higher
#   the price, the lower the chance someone buys that day.
#
#   Goal: find the pricing policy that maximizes the total
#   *expected discounted profit*, net of any overbooking costs
#   that arise if too many passengers show up on departure day.
#
# PART 1 — Solve the DP with a fixed overbooking allowance of 5.
# PART 2 — Sweep overbooking limits 5 through 20; find the best.


# --------------------------------------------------
# SECTION 1: Imports
# --------------------------------------------------

import numpy as np
from scipy.stats import binom
import matplotlib.pyplot as plt


# --------------------------------------------------
# SECTION 2: Global Parameters
# --------------------------------------------------
# All tunable numbers live here. Keeping them in one place
# makes the model easy to audit, adjust, and explain.

# --- Price options for each cabin (dollars per ticket) ---
COACH_PRICES = [300, 325, 350]
FIRST_PRICES = [425, 500]

# --- Base demand probabilities (chance of one sale per day) ---
# Lower price → higher probability. These are given by the problem.
COACH_DEMAND = {
    300: 0.65,
    325: 0.45,
    350: 0.30,
}

FIRST_DEMAND = {
    425: 0.08,
    500: 0.04,
}

# --- First-class spillover effect ---
# When first-class sells out, some would-be first-class buyers
# redirect their interest to coach. This adds a small boost
# to coach demand for the remainder of the selling period.
FIRST_SOLD_OUT_COACH_BOOST = 0.04

# --- Physical seating capacity ---
COACH_CAPACITY = 100    # seats on the plane for coach passengers
FIRST_CAPACITY = 20     # seats on the plane for first-class passengers

# --- Show-up rates (used at departure day) ---
# Not every ticket holder actually shows up. These no-show rates
# are what make overbooking a potentially profitable strategy.
COACH_SHOW_UP_PROB = 0.95
FIRST_SHOW_UP_PROB = 0.97

# --- Overbooking penalty costs ---
# When more coach passengers show up than there are seats:
#   Option A — bump them into first-class if a seat is open ($50)
#   Option B — deny boarding if first-class is also full ($425)
BUMP_TO_FIRST_COST   = 50
DENIED_BOARDING_COST = 425

# --- Daily discount factor ---
# A dollar of revenue collected today is worth slightly more than
# one collected tomorrow. We use a 17% annual rate, converted to
# a per-day multiplier via continuous-style daily compounding.
ANNUAL_DISCOUNT_RATE = 0.17
DAILY_DISCOUNT       = 1 / (1 + ANNUAL_DISCOUNT_RATE / 365)

# --- Selling horizon ---
SELLING_HORIZON = 365   # days before departure when ticket sales begin


# --------------------------------------------------
# SECTION 3: Terminal Cost Function
# --------------------------------------------------

def terminal_cost(coach_sold: int, first_sold: int) -> float:
    """
    Compute the expected overbooking cost on departure day (day 0).

    At this point, ticket sales are over. The remaining uncertainty
    is: how many passengers actually show up?

        Coach show-ups ~ Binomial(coach_sold, 0.95)
        First show-ups ~ Binomial(first_sold, 0.97)

    If more coach passengers show up than the 100 available seats,
    the airline faces a choice for each excess passenger:
        (1) Move them to an open first-class seat — costs $50
        (2) Deny them boarding entirely            — costs $425

    We compute the *expected* cost by looping over every possible
    combination of show-up counts, weighting each by its probability.

    Parameters
    ----------
    coach_sold : int  — total coach tickets sold before departure
    first_sold : int  — total first-class tickets sold before departure

    Returns
    -------
    float — expected dollar cost from overbooking on departure day
    """

    expected_cost = 0.0

    # Precompute the full probability mass function for each cabin.
    # This avoids calling scipy's binom.pmf repeatedly inside the loop.
    coach_show_up_counts = np.arange(coach_sold + 1)
    first_show_up_counts = np.arange(first_sold + 1)

    coach_probs = binom.pmf(coach_show_up_counts, coach_sold, COACH_SHOW_UP_PROB)
    first_probs = binom.pmf(first_show_up_counts, first_sold, FIRST_SHOW_UP_PROB)

    # --- Outer loop: number of coach passengers who show up ---
    for k_coach, prob_coach in enumerate(coach_probs):

        # How many coach passengers are beyond the physical seat limit?
        coach_excess = max(0, k_coach - COACH_CAPACITY)

        # If no one is turned away, this scenario costs nothing.
        # Skip the inner loop — there is nothing to compute.
        if coach_excess == 0:
            continue

        # --- Inner loop: number of first-class passengers who show up ---
        # We only reach here when there IS coach overflow.
        for k_first, prob_first in enumerate(first_probs):

            # How many first-class seats are unoccupied at departure?
            first_class_open_seats = max(0, FIRST_CAPACITY - k_first)

            # Absorb overflow into first-class, up to its available capacity
            bumped_to_first = min(coach_excess, first_class_open_seats)

            # Any remaining overflow must be denied boarding
            denied_boarding = coach_excess - bumped_to_first

            # Dollar cost for this particular show-up scenario
            scenario_cost = (bumped_to_first  * BUMP_TO_FIRST_COST
                           + denied_boarding   * DENIED_BOARDING_COST)

            # Weight by joint probability and accumulate
            expected_cost += prob_coach * prob_first * scenario_cost

    return expected_cost


# --------------------------------------------------
# SECTION 4: DP Solver Function
# --------------------------------------------------

def solve_dp(overbook_limit: int) -> float:
    """
    Solve the airline pricing problem via backward-induction DP.

    We work backwards from departure (day 0) to the first day of
    sales (day 365). At each step we ask: given where we are today,
    what price combination yields the highest expected profit from
    now until departure?

    ── STATE ──────────────────────────────────────────────────────
        (t, coach_sold, first_sold)

        t          — days remaining until departure   (0 … 365)
        coach_sold — coach tickets sold so far        (0 … 100 + overbook_limit)
        first_sold — first-class tickets sold so far  (0 … 20)

    ── BELLMAN EQUATION ───────────────────────────────────────────
        V(t, cs, fs) = max over (coach_price, first_price) of
            E[ revenue_today + γ · V(t−1, cs', fs') ]

        where cs' and fs' are updated by whatever sale(s) occur.

    ── BASE CASE ──────────────────────────────────────────────────
        V(0, cs, fs) = −terminal_cost(cs, fs)

        At t=0 there is no more revenue, only the expected cost
        of overbooking, which we subtract from profit.

    Parameters
    ----------
    overbook_limit : int
        Number of coach tickets the airline is willing to sell
        beyond the physical 100-seat capacity.

    Returns
    -------
    float
        The maximum expected discounted profit, starting from
        day 365 with zero tickets sold in either cabin.
    """

    max_coach_tickets = COACH_CAPACITY + overbook_limit
    max_first_tickets = FIRST_CAPACITY

    # ── Allocate the value table ────────────────────────────────
    # value[t, cs, fs] stores the optimal expected profit from
    # state (t, cs, fs) forward to departure.
    value = np.zeros(
        (SELLING_HORIZON + 1, max_coach_tickets + 1, max_first_tickets + 1)
    )

    # ── Base case: fill in departure-day values (t = 0) ─────────
    # No revenue is earned on departure day — only overbooking costs.
    # We precompute all (cs, fs) combinations up front for clarity.
    print(f"  [overbook={overbook_limit:>2}]  Precomputing departure-day costs ...",
          end="\r")

    for cs in range(max_coach_tickets + 1):
        for fs in range(max_first_tickets + 1):
            # Terminal value is negative because overbooking is a cost
            value[0, cs, fs] = -terminal_cost(cs, fs)

    # ── Backward induction: day 1 through day 365 ───────────────
    # We move backward through time, filling in the value table one
    # day at a time. Each day's optimal value depends on the values
    # we already computed for the following day (t − 1).
    for t in range(1, SELLING_HORIZON + 1):

        if t % 73 == 0 or t == SELLING_HORIZON:
            pct = int(100 * t / SELLING_HORIZON)
            print(f"  [overbook={overbook_limit:>2}]  Day {t:>3} / {SELLING_HORIZON}  ({pct}%) ...",
                  end="\r")

        for cs in range(max_coach_tickets + 1):
            for fs in range(max_first_tickets + 1):

                # Determine whether each cabin can still accept tickets
                can_sell_coach = (cs < max_coach_tickets)
                can_sell_first = (fs < max_first_tickets)

                # When first-class is sold out, some buyers redirect to coach.
                # We apply the demand boost regardless of the price choice.
                first_class_is_full = not can_sell_first

                # ── Index helpers ────────────────────────────────
                # We need the next state if a sale happens.
                # Use cs/fs unchanged when a cabin is full — the 0
                # sale probability ensures that term won't contribute,
                # and this avoids any out-of-bounds index access.
                next_cs = (cs + 1) if can_sell_coach else cs
                next_fs = (fs + 1) if can_sell_first else fs

                best_expected_profit = -np.inf

                # ── Try every price combination ──────────────────
                for coach_price in COACH_PRICES:
                    for first_price in FIRST_PRICES:

                        # Start with the base demand probabilities
                        p_coach = COACH_DEMAND[coach_price]
                        p_first = FIRST_DEMAND[first_price]

                        # Apply spillover boost when first-class is sold out
                        if first_class_is_full:
                            p_coach = min(1.0, p_coach + FIRST_SOLD_OUT_COACH_BOOST)

                        # A full cabin cannot generate a sale, regardless of price
                        if not can_sell_coach:
                            p_coach = 0.0
                        if not can_sell_first:
                            p_first = 0.0

                        # ── Four outcomes for today ──────────────────────────
                        # Each day, 0 or 1 coach ticket and 0 or 1 first-class
                        # ticket can be sold. These events are independent.

                        # Outcome 1 — No sale in either cabin
                        prob_no_sale = (1 - p_coach) * (1 - p_first)
                        val_no_sale  = (DAILY_DISCOUNT
                                        * value[t - 1, cs, fs])

                        # Outcome 2 — Coach ticket sold, first-class not
                        prob_coach_only = p_coach * (1 - p_first)
                        val_coach_only  = (coach_price
                                           + DAILY_DISCOUNT
                                           * value[t - 1, next_cs, fs])

                        # Outcome 3 — First-class ticket sold, coach not
                        prob_first_only = (1 - p_coach) * p_first
                        val_first_only  = (first_price
                                           + DAILY_DISCOUNT
                                           * value[t - 1, cs, next_fs])

                        # Outcome 4 — Both tickets sold on the same day
                        prob_both_sold = p_coach * p_first
                        val_both_sold  = (coach_price + first_price
                                          + DAILY_DISCOUNT
                                          * value[t - 1, next_cs, next_fs])

                        # ── Expected profit from this price decision ─────────
                        expected_profit = (prob_no_sale    * val_no_sale
                                         + prob_coach_only * val_coach_only
                                         + prob_first_only * val_first_only
                                         + prob_both_sold  * val_both_sold)

                        # Keep the best pricing decision seen so far
                        if expected_profit > best_expected_profit:
                            best_expected_profit = expected_profit

                # Store the best achievable value for this state
                value[t, cs, fs] = best_expected_profit

    print(f"  [overbook={overbook_limit:>2}]  Complete.{' ' * 50}")

    # The answer is the value at the very first state:
    # 365 days to go, no tickets sold yet in either cabin.
    return value[SELLING_HORIZON, 0, 0]


# --------------------------------------------------
# SECTION 5: Overbooking Optimization Loop (Part 2)
# --------------------------------------------------

def find_best_overbooking_policy() -> dict:
    """
    Compare overbooking policies by solving the DP for limits 5 to 20.

    More overbooking means more potential revenue (we sell more tickets)
    but also higher expected costs at departure. This function finds the
    level where the net benefit is highest.

    Returns
    -------
    dict with keys:
        'limits'      — list of overbooking limits tested
        'profits'     — expected profit for each limit
        'best_limit'  — the limit that produced the highest profit
        'best_profit' — the corresponding expected profit
    """

    overbook_limits  = list(range(5, 21))   # 5, 6, 7, ..., 20
    expected_profits = []

    print("\n" + "=" * 60)
    print("  PART 2 — Searching for the Optimal Overbooking Policy")
    print("=" * 60)
    print(f"\n  {'Overbook Limit':>16}  |  {'Expected Profit':>18}")
    print("  " + "-" * 42)

    for limit in overbook_limits:
        profit = solve_dp(limit)
        expected_profits.append(profit)
        print(f"  {limit:>16}  |  ${profit:>17,.2f}")

    print("  " + "=" * 42)

    # Identify the winner
    best_index  = int(np.argmax(expected_profits))
    best_limit  = overbook_limits[best_index]
    best_profit = expected_profits[best_index]

    return {
        'limits':      overbook_limits,
        'profits':     expected_profits,
        'best_limit':  best_limit,
        'best_profit': best_profit,
    }


def plot_overbooking_results(results: dict) -> None:
    """
    Visualize expected profit as a function of overbooking limit.

    A well-designed chart here tells the story immediately:
      - Is there a clear peak, or does profit plateau?
      - How steep is the penalty for over- or under-booking?
      - Where exactly should the airline draw the line?
    """

    limits      = results['limits']
    profits     = results['profits']
    best_limit  = results['best_limit']
    best_profit = results['best_profit']

    fig, ax = plt.subplots(figsize=(10, 6))

    # --- Main profit curve ---
    ax.plot(limits, profits,
            marker='o', markersize=8, linewidth=2.5,
            color='steelblue', label='Expected Discounted Profit')

    # --- Mark and label the optimal point ---
    ax.scatter([best_limit], [best_profit],
               color='tomato', s=140, zorder=6, label=f'Best limit = {best_limit}')

    ax.axvline(x=best_limit, color='tomato',
               linestyle='--', linewidth=1.6, alpha=0.7)

    # Annotate the peak with its dollar value
    ax.annotate(
        f"  ${best_profit:,.0f}\n  (limit = {best_limit})",
        xy=(best_limit, best_profit),
        xytext=(best_limit + 0.4, best_profit),
        fontsize=10,
        color='tomato',
        va='center'
    )

    # --- Formatting ---
    ax.set_title(
        'Expected Discounted Profit vs. Overbooking Limit',
        fontsize=14, fontweight='bold', pad=14
    )
    ax.set_xlabel(
        'Coach Overbooking Limit (extra tickets sold beyond 100 seats)',
        fontsize=12
    )
    ax.set_ylabel('Expected Discounted Profit ($)', fontsize=12)
    ax.legend(fontsize=11, loc='lower right')
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda val, _: f'${val:,.0f}')
    )

    plt.tight_layout()
    plt.savefig('overbooking_optimization.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("\n  Chart saved → overbooking_optimization.png")


# --------------------------------------------------
# SECTION 6: Main Execution Block
# --------------------------------------------------

def main():
    """
    Run the full Parts 1 and 2 analysis end-to-end.

    ── PART 1 ──────────────────────────────────────────────────────
    Establish a baseline by solving the DP with overbooking = 5.
    This tells us: what is the best we can do if we allow 5 extra
    coach tickets to be sold?

    ── PART 2 ──────────────────────────────────────────────────────
    Expand the search. Solve the DP for overbooking limits 5 through
    20. Identify the level that maximizes expected discounted profit,
    and produce a chart showing the full trade-off curve.

    ── RUNTIME NOTE ────────────────────────────────────────────────
    Each DP solve covers ~365 days × 121 coach states × 21 first-
    class states = ~930,000 states. With 16 runs for Part 2, expect
    this script to take several minutes. Progress is printed as each
    run advances.
    """

    print("\n" + "=" * 60)
    print("  MSBA Optimization 2  |  Project 2")
    print("  Airline Ticket Pricing via Dynamic Programming")
    print("=" * 60)

    # ── PART 1: Baseline with overbooking limit = 5 ─────────────
    print("\n  PART 1: Solving the DP with a fixed overbooking limit of 5.")
    print("  This is our benchmark — 5 seats beyond the 100-seat coach cabin.\n")

    baseline_profit = solve_dp(overbook_limit=5)

    print(f"\n  ┌─────────────────────────────────────────────┐")
    print(f"  │  PART 1 RESULT                              │")
    print(f"  │  Overbooking Limit    :  5 extra seats      │")
    print(f"  │  Expected Profit      : ${baseline_profit:>12,.2f}         │")
    print(f"  └─────────────────────────────────────────────┘")

    # ── PART 2: Optimize over limits 5 through 20 ───────────────
    print("\n  PART 2: Sweeping overbooking limits from 5 to 20.")
    print("  Each run is a full DP solve. This will take a few minutes.\n")

    results = find_best_overbooking_policy()

    print(f"\n  ┌─────────────────────────────────────────────┐")
    print(f"  │  PART 2 RESULT                              │")
    print(f"  │  Best Overbooking Limit : {results['best_limit']:>2} extra seats     │")
    print(f"  │  Best Expected Profit   : ${results['best_profit']:>12,.2f}         │")
    print(f"  └─────────────────────────────────────────────┘")

    plot_overbooking_results(results)

    print("\n  Analysis complete.\n")


if __name__ == "__main__":
    main()

# --------------------------------------------------
# SECTION 7: Task 3 & 4 (Flexible Policy & Sensitivity)
# --------------------------------------------------

def solve_dp_flexible(max_coach=130):
    """
    Task 3: Solve DP where 'No Sale' is an option.
    Hard cap is fixed at 130, but the airline can choose p=0 for coach.
    """
    max_first = FIRST_CAPACITY
    # value[t, coach_sold, first_sold]
    value = np.zeros((SELLING_HORIZON + 1, max_coach + 1, max_first + 1))

    # Base Case (t=0): terminal_cost logic remains the same
    for cs in range(max_coach + 1):
        for fs in range(max_first + 1):
            value[0, cs, fs] = -terminal_cost(cs, fs)

    # Backward Induction
    for t in range(1, SELLING_HORIZON + 1):
        if t % 100 == 0:
            print(f"    [Flexible Policy] Processing Day {t}...", end="\r")
            
        for cs in range(max_coach + 1):
            for fs in range(max_first + 1):
                
                # Actions: (Price, Probability)
                # Task 3 adds (0, 0.0) as a strategic choice
                coach_actions = [(300, 0.65), (325, 0.45), (350, 0.30), (0, 0.0)]
                first_actions = [(425, 0.08), (500, 0.04)]
                
                best_ev = -np.inf
                
                for cp, cq in coach_actions:
                    for fp, fq in first_actions:
                        
                        # Apply First-Class sold-out boost (4 percentage points)
                        p_c = cq
                        if fs == max_first and cp > 0:
                            p_c = min(1.0, cq + 0.04)
                        
                        p_f = fq
                        
                        # Boundary checks: if we physically hit 130 or 20, prob is 0
                        if cs >= max_coach: p_c = 0.0
                        if fs >= max_first: p_f = 0.0
                        
                        # Future states
                        v_none = value[t-1, cs, fs]
                        v_c    = value[t-1, min(cs+1, max_coach), fs]
                        v_f    = value[t-1, cs, min(fs+1, max_first)]
                        v_both = value[t-1, min(cs+1, max_coach), min(fs+1, max_first)]
                        
                        # Expected Value calculation (same logic as group mate)
                        ev = ( (1-p_c)*(1-p_f) * (DAILY_DISCOUNT * v_none) +
                               p_c*(1-p_f)     * (cp + DAILY_DISCOUNT * v_c) +
                               (1-p_c)*p_f     * (fp + DAILY_DISCOUNT * v_f) +
                               p_c*p_f         * (cp + fp + DAILY_DISCOUNT * v_both) )
                        
                        if ev > best_ev:
                            best_ev = ev
                            
                value[t, cs, fs] = best_ev
                
    return value[SELLING_HORIZON, 0, 0]

def run_sensitivity_analysis(best_limit, multipliers):
    """
    Task 4: Re-solve the Part 2 DP with shifted demand probabilities.
    """
    results = []
    # Store original demand to restore after loop
    orig_coach = COACH_DEMAND.copy()
    orig_first = FIRST_DEMAND.copy()

    print(f"\n  {'Multiplier':>12} | {'Expected Profit':>18}")
    print("  " + "-" * 35)

    for m in multipliers:
        # Update Global Demands for the solver
        for p in COACH_DEMAND: COACH_DEMAND[p] = orig_coach[p] * m
        for p in FIRST_DEMAND: FIRST_DEMAND[p] = orig_first[p] * m
        
        # We use the best limit found in Step 2
        profit = solve_dp(best_limit)
        results.append((m, profit))
        print(f"  {m:>12.2f} | ${profit:>17,.2f}")

    # Restore original values
    for p in orig_coach: COACH_DEMAND[p] = orig_coach[p]
    for p in orig_first: FIRST_DEMAND[p] = orig_first[p]
    
    return results

def plot_task_4(sensitivity_results):
    mults, profits = zip(*sensitivity_results)
    plt.figure(figsize=(10, 5))
    plt.plot(mults, profits, marker='D', color='darkred', linewidth=2)
    plt.title('Task 4: Sensitivity of Profit to Demand Fluctuations', fontweight='bold')
    plt.xlabel('Demand Multiplier (1.0 = Baseline)')
    plt.ylabel('Expected Profit ($)')
    plt.grid(True, alpha=0.3)
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x:,.0f}'))
    plt.show()

def main_v2():
    # 1. Run Baseline (Assuming Task 2 found limit 15 as an example)
    print("\n" + "="*60)
    print("  AIRLINE OPTIMIZATION: TASKS 3 & 4")
    print("="*60)
    
    # Run Task 3
    print("\n  TASK 3: Solving Flexible 'No Sale' Policy (Limit 130)...")
    flex_profit = solve_dp_flexible(max_coach=130)
    
    # We need a Task 2 result for comparison (using limit 15 as placeholder)
    task2_best_limit = 15 
    task2_best_profit = solve_dp(task2_best_limit)
    
    print(f"\n  [Task 3 Result] Flexible Policy Profit: ${flex_profit:,.2f}")
    print(f"  [Comparison]    Hard Cap (15) Profit:  ${task2_best_profit:,.2f}")
    
    if flex_profit > task2_best_profit:
        print("  DECISION: Flexible Policy is SUPERIOR.")
    else:
        print("  DECISION: Hard Cap Policy is SUPERIOR.")

    # Run Task 4
    print("\n  TASK 4: Running Sensitivity Analysis on Sales Probabilities...")
    multipliers = [0.90, 0.95, 1.0, 1.05, 1.10]
    sens_results = run_sensitivity_analysis(task2_best_limit, multipliers)
    
    plot_task_4(sens_results)
    print("\n  Analysis Complete.")

if __name__ == "__main__":
    main_v2()
