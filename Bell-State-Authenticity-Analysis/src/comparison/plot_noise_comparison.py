
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from qiskit.quantum_info import Statevector, DensityMatrix

from src.bell_states.bell_state_preparation import create_bell_state
from src.noise_models.depolarizing_noise import apply_global_depolarizing_noise

from src.comparison.noise_comparison import (
    calculate_max_chsh,
    calculate_qst_fidelity,
    calculate_visibility
)


def generate_results():

    bell_states = [
        "phi_plus",
        "phi_minus",
        "psi_plus",
        "psi_minus"
    ]

    noise_levels = np.linspace(0, 1, 21)

    results = []

    for state in bell_states:

        circuit = create_bell_state(state)
        sv = Statevector.from_instruction(circuit)
        ideal_rho = DensityMatrix(sv).data

        for p in noise_levels:

            noisy_rho = apply_global_depolarizing_noise(
                ideal_rho, p
            )

            chsh = calculate_max_chsh(noisy_rho)

            fidelity = calculate_qst_fidelity(
                noisy_rho, ideal_rho
            )

            visibility = calculate_visibility(noisy_rho)

            results.append({
                "Bell State": state,
                "Noise": p,
                "CHSH": chsh,
                "QST Fidelity": fidelity,
                "Visibility": visibility
            })

    return pd.DataFrame(results)


def plot_results(df):

    os.makedirs("results/plots", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)

    # Save numerical results
    df.to_csv(
        "data/processed/noise_comparison.csv",
        index=False
    )

    # Plot each measure
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    measures = [
        ("CHSH", "Maximum CHSH Value"),
        ("QST Fidelity", "QST Fidelity"),
        ("Visibility", "Visibility")
    ]

    for ax, (column, title) in zip(axes, measures):

        for state in df["Bell State"].unique():

            subset = df[df["Bell State"] == state]

            ax.plot(
                subset["Noise"],
                subset[column],
                marker="o",
                markersize=3,
                label=state
            )

        ax.set_xlabel("Depolarizing Noise Probability")
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend()

        if column == "CHSH":
            ax.axhline(
                y=2,
                linestyle="--",
                label="CHSH limit"
            )

    plt.tight_layout()

    plt.savefig(
        "results/plots/noise_comparison.png",
        dpi=300
    )

    plt.show()


if __name__ == "__main__":

    df = generate_results()

    print("\nFirst few rows of results:")
    print(df.head(10).to_string(index=False))

    plot_results(df)

    print("\nResults saved successfully:")
    print("CSV: data/processed/noise_comparison.csv")
    print("Plot: results/plots/noise_comparison.png")
