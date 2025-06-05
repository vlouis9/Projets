import streamlit as st
import pandas as pd

def main():
    st.title("Mercato Squad Builder")
    st.markdown(
        """
        This app builds your Mercato squad from an uploaded Excel file that has players’ information.
        
        **Instructions:**
        - Prepare an Excel file with at least these columns: `Joueur`, `Poste`, `Club`, `Cote`.
        - Use the sidebar to choose your target formation, total squad size, budget, and customize the delta multipliers.
        
        The app will then select your CORE (starting XI) based on the formation and complete your bench to ensure the overall MPG \
        minimums (2 GK, 6 defenders, 6 midfielders, 4 forwards) are met. Extra players (if your squad size is larger than 18) are added to \
        the bench with priority.
        """
    )
    st.markdown("*Code available on GitHub: [Github repo link](https://github.com/yourusername/mercato-squad-builder)*")
    
    # --- Sidebar options --- #
    st.sidebar.header("Squad Configuration")
    formation = st.sidebar.selectbox("Select Formation", options=["3-4-3", "4-3-3", "4-4-2"])
    total_squad = st.sidebar.number_input("Total Squad Size (min 18)", min_value=18, value=18, step=1)
    budget = st.sidebar.number_input("Total Budget (M€)", min_value=1.0, value=500.0, step=1.0)
    
    st.sidebar.header("Delta Multipliers")
    core_multiplier = st.sidebar.slider("Core Multiplier", min_value=1.0, max_value=2.0, value=1.35, step=0.05,
                                        help="Multiplier for recommended prices of CORE players (recommended: ~1.3-1.5)")
    bench_multiplier = st.sidebar.slider("Bench Multiplier", min_value=1.0, max_value=2.0, value=1.15, step=0.05,
                                         help="Multiplier for recommended prices of bench players (recommended: ~1.1-1.3)")
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Note:** The recommended prices for bench players are then scaled so that the overall squad cost exactly equals the budget.")
    
    # --- File upload --- #
    uploaded_file = st.file_uploader("Upload player Excel file", type=["xlsx"])
    if uploaded_file is not None:
        try:
            # Read the file and ensure necessary columns exist.
            df = pd.read_excel(uploaded_file)
            required_cols = ["Joueur", "Poste", "Club", "Cote"]
            if not all(col in df.columns for col in required_cols):
                st.error("The uploaded file must contain the columns: Joueur, Poste, Club, and Cote.")
                return
            
            st.success("File uploaded successfully!")
            st.write("Preview of data:", df.head())
            
            # --- Define Category from Poste --- #
            # MPG grouping: Goalkeepers ("G"); Defenders (if Poste is "DC" or "DL"); Midfielders ("MD" or "MO"); Forwards ("A")
            def get_category(poste):
                if poste == "G":
                    return "G"
                elif poste in ["DC", "DL"]:
                    return "D"
                elif poste in ["MD", "MO"]:
                    return "M"
                elif poste == "A":
                    return "A"
                else:
                    return "Other"
            
            df["Category"] = df["Poste"].apply(get_category)
            
            # --- Determine CORE XI counts based on formation ---
            # Here we assume the starting XI always has 11 players:
            # Examples:
            #   3-4-3 => 1 GK, 3 defenders, 4 midfielders, 3 forwards
            #   4-3-3 => 1 GK, 4 defenders, 3 midfielders, 3 forwards
            #   4-4-2 => 1 GK, 4 defenders, 4 midfielders, 2 forwards
            
            if formation == "3-4-3":
                core_counts = {"G": 1, "D": 3, "M": 4, "A": 3}
            elif formation == "4-3-3":
                core_counts = {"G": 1, "D": 4, "M": 3, "A": 3}
            elif formation == "4-4-2":
                core_counts = {"G": 1, "D": 4, "M": 4, "A": 2}
            else:
                st.error("Formation not supported.")
                return
            
            # --- Overall MPG minimums (entire squad) --- #
            overall_min = {"G": 2, "D": 6, "M": 6, "A": 4}
            
            # --- CORE selection: for each category, select the top players by Cote ---
            core_selection = []
            for cat, num in core_counts.items():
                candidates = df[df["Category"] == cat].sort_values(by="Cote", ascending=False)
                if len(candidates) < num:
                    st.error(f"Not enough players in category {cat} to fill CORE requirements.")
                    return
                selected = candidates.head(num).copy()
                selected["Role"] = "CORE"
                core_selection.append(selected)
            core_df = pd.concat(core_selection)
            
            # Remove CORE players from further consideration.
            remaining_df = df.drop(core_df.index)
            
            # --- Minimal bench selection: ensure overall MPG minimums are met ---
            bench_required = {}
            for cat, min_req in overall_min.items():
                current = core_df[core_df["Category"] == cat].shape[0]
                bench_required[cat] = max(0, min_req - current)
            
            bench_parts = []
            for cat, num in bench_required.items():
                if num > 0:
                    candidates = remaining_df[remaining_df["Category"] == cat].sort_values(by="Cote", ascending=False)
                    if len(candidates) < num:
                        st.error(f"Not enough players in category {cat} to meet minimum bench requirement.")
                        return
                    selected = candidates.head(num).copy()
                    selected["Role"] = "Bench"
                    bench_parts.append(selected)
            bench_df = pd.concat(bench_parts) if bench_parts else pd.DataFrame(columns=df.columns)
            
            # Remove the minimal bench selections from remaining players.
            remaining_df = remaining_df.drop(bench_df.index)
            
            # --- Allocate extra bench spots if the total squad size > minimal squad ---
            # Minimal squad size is CORE + minimal bench.
            minimal_squad_count = core_df.shape[0] + bench_df.shape[0]
            extra_bench_needed = total_squad - minimal_squad_count
            extra_bench_df = pd.DataFrame()
            if extra_bench_needed > 0:
                # For extra bench picks, select the remaining highest-Cote players (regardless of category)
                extra_bench_df = remaining_df.sort_values(by="Cote", ascending=False).head(extra_bench_needed).copy()
                extra_bench_df["Role"] = "Bench"
            
            # --- Final Squad ---
            squad_df = pd.concat([core_df, bench_df, extra_bench_df])
            
            st.subheader("Squad Selection")
            st.write(squad_df[["Joueur", "Poste", "Category", "Club", "Cote", "Role"]])
            
            # --- Compute Recommended Prices ---
            # CORE players: recommended price = cote * core_multiplier
            # Bench players: recommended price = cote * bench_multiplier, then scaled so that:
            #   Total squad cost = (sum(core prices)) + (sum(bench prices adjusted)) equals the budget.
            # This ensures that bench prices are increased (or decreased) proportionally – with bench being the “price lever.”
            
            squad_df["Prix_temp"] = squad_df.apply(
                lambda row: row["Cote"] * core_multiplier if row["Role"] == "CORE" else row["Cote"] * bench_multiplier,
                axis=1,
            )
            
            # Sum CORE recommended prices.
            core_sum = squad_df[squad_df["Role"] == "CORE"]["Prix_temp"].sum()
            bench_target = budget - core_sum
            bench_sum_temp = squad_df[squad_df["Role"] == "Bench"]["Prix_temp"].sum()
            
            bench_scaling_factor = bench_target / bench_sum_temp if bench_sum_temp != 0 else 1.0
            
            def compute_price(row):
                if row["Role"] == "CORE":
                    return row["Cote"] * core_multiplier
                else:
                    return row["Cote"] * bench_multiplier * bench_scaling_factor
            
            squad_df["Prix_recommande"] = squad_df.apply(compute_price, axis=1)
            total_cost = squad_df["Prix_recommande"].sum()
            
            st.subheader("Final Squad with Recommended Prices")
            st.write(
                squad_df[
                    ["Joueur", "Poste", "Category", "Club", "Cote", "Role", "Prix_recommande"]
                ]
            )
            st.write(f"**Total Cost: {total_cost:.2f} M€ (Target Budget: {budget:.2f} M€)**")
            
        except Exception as e:
            st.error(f"An error occurred while processing the file: {e}")
    else:
        st.info("Please upload an Excel file to begin.")

if __name__ == "__main__":
    main()
