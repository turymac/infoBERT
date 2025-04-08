import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from utils.utils import extract_ckpt_number

def avg_distance_per_class(df, tag): # PLOT 1

    # Colonne numeriche dei modelli (escludendo categoria e label finale)
    epoch_cols = [col for col in df.columns if col not in ["category", "environment_info", "sentence"]]

    epoch_cols = sorted(epoch_cols, key=extract_ckpt_number)
    x_labels = sorted([f"epoch-{i}" for i in range(len(epoch_cols)-1)] + ["baseline"], key=extract_ckpt_number)

    # Separazione delle classi
    df_Y = df[df["environment_info"] == 1]
    df_N = df[df["environment_info"] == 0]

    # Calcolo della media per ogni epoca per ciascuna classe
    mean_Y = df_Y[epoch_cols].mean()
    mean_N = df_N[epoch_cols].mean()

    # Grafico
    plt.figure(figsize=(10, 6))
    plt.plot(epoch_cols, mean_Y, marker='o', label="Y", color="green")
    plt.plot(epoch_cols, mean_N, marker='o', label="N", color="red")

    plt.title("Avg distance per class across epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Avg Distance from closest centroid")
    plt.xticks(ticks=epoch_cols, labels=x_labels, rotation=45)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(f"plots/{tag}_avg_distance_per_class.png", dpi=300, bbox_inches='tight')
    plt.show()

def distance_boxplot(df, tag): # PLOT 2

    # Colonne delle epoche
    epoch_cols = [col for col in df.columns if col not in ["category", "environment_info", "sentence"]]

    epoch_cols = sorted(epoch_cols, key=extract_ckpt_number)
    x_labels = sorted([f"epoch-{i}" for i in range(len(epoch_cols) - 1)] + ["baseline"], key=extract_ckpt_number)

    # Ricostruiamo un DataFrame "long" per facilitare il boxplot
    data_long = pd.melt(
        df,
        id_vars=["environment_info"],
        value_vars=epoch_cols,
        var_name="epoca",
        value_name="distanza"
    )

    # Grafico
    plt.figure(figsize=(14, 6))
    sns.boxplot(
        x="epoca",
        y="distanza",
        hue="environment_info",
        data=data_long,
        palette={0: "orange", 1: "skyblue"}
    )

    plt.title("Distribution of distances across epochs per class")
    plt.xlabel("Epoch")
    plt.ylabel("Distance")
    plt.xticks(ticks=epoch_cols, labels=x_labels, rotation=45)
    plt.legend(title="Class")
    plt.grid(True, axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()

    plt.savefig(f"plots/{tag}_distance_boxplot.png", dpi=300, bbox_inches='tight')
    plt.show()

def distance_difference_heatmap(df, tag): # PLOT 3

    # Colonne delle epoche
    epoch_cols = [col for col in df.columns if col not in ["category", "environment_info", "sentence"]]
    x_labels = sorted([f"epoch-{i+1}" for i in range(len(epoch_cols)-1)] + ["baseline"], key=extract_ckpt_number)

    epoch_cols = sorted(epoch_cols, key=extract_ckpt_number)

    # Calcola le medie per classe
    mean_Y = df[df["environment_info"] == 1][epoch_cols].mean()
    mean_N = df[df["environment_info"] == 0][epoch_cols].mean()

    # Calcola la differenza assoluta
    diff = (mean_Y - mean_N).abs().to_frame().T  # Riga unica

    # Crea heatmap
    plt.figure(figsize=(12, 4))
    sns.heatmap(
        diff,
        annot=True,
        fmt=".2f",
        cmap="YlOrRd",
        cbar_kws={"label": "Differenza media Y - N"}
    )

    plt.title("Absolute difference between class distances across epochs")
    plt.xlabel("Epoch")
    plt.xticks(ticks=range(len(epoch_cols)), labels=x_labels, rotation=45)
    plt.yticks([], [])  # Niente etichette sull'asse Y
    plt.tight_layout()

    plt.savefig(f"plots/{tag}_distance_difference_heatmap.png", dpi=300, bbox_inches='tight')
    plt.show()

def distance_variance_plot(df, tag): # PLOT 4

    # Colonne delle epoche
    epoch_cols = [col for col in df.columns if col not in ["category", "environment_info", "sentence"]]
    x_labels = sorted([f"epoch-{i + 1}" for i in range(len(epoch_cols) - 1)] + ["baseline"], key=extract_ckpt_number)
    epoch_cols = sorted(epoch_cols, key=extract_ckpt_number)

    # Calcola varianza per ciascuna epoca e classe
    var_Y = df[df["environment_info"] == 1][epoch_cols].var()
    var_N = df[df["environment_info"] == 0][epoch_cols].var()

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(epoch_cols, var_Y, marker='o', label="Info", color="green")
    plt.plot(epoch_cols, var_N, marker='o', label="No Info", color="red")

    plt.title("Distance variance across epochs per class")
    plt.xlabel("Epoch")
    plt.ylabel("Variance")
    plt.xticks(ticks=epoch_cols, labels=x_labels, rotation=45)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(f"plots/{tag}_distance_variance.png", dpi=300, bbox_inches='tight')
    plt.show()

def plot_accuracy_per_category(path, tag, metric):

    df = pd.read_excel(path)
    # Imposta 'category' come indice per comodità
    df.set_index("category", inplace=True)

    # Lista delle epoche ordinate
    epoch_cols = [col for col in df.columns if col not in ["category"]]
    x_labels = sorted([f"epoch-{i + 1}" for i in range(len(epoch_cols) - 1)] + ["baseline"], key=extract_ckpt_number)

    epoch_cols = sorted(epoch_cols, key=extract_ckpt_number)

    # Estrai media (se presente)
    avg_row = df.loc["Avg"] if "Avg" in df.index else df[epoch_cols].mean()

    # Rimuove la riga 'Avg' dal dataset delle singole categorie
    df_no_avg = df.drop(index="Avg", errors="ignore")

    # Plot
    plt.figure(figsize=(12, 6))

    # Linee per ogni categoria
    for category in df_no_avg.index:
        plt.plot(epoch_cols, df_no_avg.loc[category, epoch_cols], marker='o', label=category)

    # Linea media ben visibile
    if avg_row is not None:
        plt.plot(
            epoch_cols,
            avg_row[epoch_cols],
            label="Avg",
            color="black",
            linestyle="--",
            linewidth=2.5,
            marker='X',
            markersize=8
        )

    plt.title(f"{metric} @ epoch")
    plt.xlabel("Epoch")
    plt.ylabel(f"{metric}")
    plt.xticks(ticks=epoch_cols, labels=x_labels, rotation=45)
    plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(f"plots/{tag}_{metric}_per_category.png", dpi=300, bbox_inches='tight')
    plt.show()

def accuracy_heatmap(path, tag, metric):
    df = pd.read_excel(path)
    # Imposta 'category' come indice
    df.set_index("category", inplace=True)

    # Colonne delle epoche
    epoch_cols = [col for col in df.columns if col not in ["category"]]
    x_labels = sorted([f"epoch-{i + 1}" for i in range(len(epoch_cols) - 1)] + ["baseline"], key=extract_ckpt_number)

    epoch_cols = sorted(epoch_cols, key=extract_ckpt_number)

    # Crea la heatmap (include anche Avg se presente)
    plt.figure(figsize=(12, 6))
    sns.heatmap(
        df[epoch_cols],
        annot=True,
        fmt=".3f",
        cmap="YlGnBu",
        linewidths=0.5,
        cbar_kws={"label": metric}
    )

    plt.title(f"{metric} @ epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Category")
    plt.xticks(ticks=range(len(epoch_cols)), labels=x_labels, rotation=45)
    plt.tight_layout()

    plt.savefig(f"plots/{tag}_{metric}_heatmap.png", dpi=300, bbox_inches='tight')
    plt.show()

def best_epoch_barplot(filepath, metric, tag):
    # Carica il file Excel
    df = pd.read_excel(filepath)

    # Colonne delle epoche
    epoch_cols = [col for col in df.columns if col not in ["category"]]
    x_labels = sorted([f"epoch-{i + 1}" for i in range(len(epoch_cols) - 1)] + ["baseline"], key=extract_ckpt_number)

    epoch_cols = sorted(epoch_cols, key=extract_ckpt_number)

    # Rimuove la riga 'Avg' se presente
    df = df[df["category"] != "Avg"]

    # Trova l'epoca con la massima accuracy per ciascuna categoria
    df["best_epoch"] = df[epoch_cols].idxmax(axis=1)

    # Conta quante volte ogni epoca è risultata la migliore
    best_epoch_counts = df["best_epoch"].value_counts().sort_index()

    # Plot
    plt.figure(figsize=(10, 5))
    best_epoch_counts.plot(kind="bar", color="mediumseagreen")

    plt.title(f"Distribution of best {metric} across epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Count")
    plt.xticks(ticks=range(len(epoch_cols)), labels=x_labels, rotation=45)
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()

    plt.savefig(f"plots/{tag}_best_{metric}_barplot.png", dpi=300, bbox_inches='tight')
    plt.show()

def main():

    model = "paraphrase-mpnet-base-v2"
    ft_dt = "m-v1"
    dt = "v2"
    tag = f"{ft_dt}_{model}_{dt}"

    # Plots on distances
    # df = pd.read_excel(f"results/{tag}_ckp_distances.xlsx")
    #
    # avg_distance_per_class(df, tag)
    # distance_boxplot(df, tag)
    # distance_difference_heatmap(df, tag)
    # distance_variance_plot(df, tag)

    # # Plots on accuracy
    plot_accuracy_per_category(f"results/{tag}_ckp_accuracies.xlsx", metric="accuracy", tag=tag)
    accuracy_heatmap(f"results/{tag}_ckp_accuracies.xlsx", metric="accuracy", tag=tag)
    best_epoch_barplot(f"results/{tag}_ckp_accuracies.xlsx", metric="accuracy", tag=tag)

    # Plots on correlation
    plot_accuracy_per_category(f"results/{tag}_ckp_correlations.xlsx", metric="correlation", tag=tag)
    accuracy_heatmap(f"results/{tag}_ckp_correlations.xlsx", metric="correlation", tag=tag)
    best_epoch_barplot(f"results/{tag}_ckp_correlations.xlsx", metric="correlation", tag=tag)

if __name__ == "__main__":
    main()