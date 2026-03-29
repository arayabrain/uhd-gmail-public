"""visualize preprocessing procedures (Fig. 1)"""

import json
from pathlib import Path

import numpy as np
import torch
from mne.filter import filter_data
from scipy.stats import zscore
from termcolor import cprint

from plot_figures.make_preproc_files import preproc
from plot_figures.src.default_plt import cm_to_inch, plt

DATA_ROOT = "data/"


def check_file_existence(subject: str, date: str) -> None:
    """check if the files exist

    Args:
        subject (str): subject name
        date (str): date of the experiment
    """
    metadata = json.load(open(Path(DATA_ROOT) / subject / date / "metadata.json"))
    session_names = [key for key in list(metadata.keys()) if key != "subject"]
    for session_name in session_names:
        cprint(session_name, "cyan")
        before_preproc_paths = list(
            (
                Path(DATA_ROOT)
                / subject
                / date
                / "eeg_before_preproc"
                / f"{date}_{session_name}"
            ).glob("*.npy")
        )
        margin_before_preproc_paths = list(
            (
                Path(DATA_ROOT)
                / subject
                / date
                / "eeg_margin_before_preproc"
                / f"{date}_{session_name}"
            ).glob("*.npy")
        )
        after_preproc_paths = list(
            (
                Path(DATA_ROOT)
                / subject
                / date
                / "eeg_after_preproc"
                / f"{date}_{session_name}"
            ).glob("*.pt")
        )
        cprint(f"before_preproc: {len(before_preproc_paths)}", "cyan")
        cprint(f"margin_before_preproc: {len(margin_before_preproc_paths)}", "cyan")
        cprint(f"after_preproc: {len(after_preproc_paths)}", "cyan")
        if (
            len(before_preproc_paths)
            == len(margin_before_preproc_paths)
            == len(after_preproc_paths)
        ):
            cprint("OK", "green")
        if len(before_preproc_paths) > 0:
            eeg = np.load(list(before_preproc_paths)[0])
            cprint(f"shape of before_preproc: {eeg.shape}\n", "cyan")


def plot_speech_waveform(
    subject: str, date: str, session_name: str, save_path: Path, task: str
) -> None:
    """plot speech waveform

    Args:
        subject (str): subject name
        date (str): date of the experiment
        session_name (str): session name
        save_path (Path): path to save the figure
        task (str): task name
    """
    before_preproc_paths = list(
        (
            Path(DATA_ROOT)
            / subject
            / date
            / "eeg_before_preproc"
            / f"{date}_{session_name}"
        ).glob("*.npy")
    )
    eeg = np.load(before_preproc_paths[2])
    speech = eeg[130, :] - eeg[131, :]
    speech = filter_data(speech, 256, 100, 127)

    fig, ax = plt.subplots(figsize=(1.25 * cm_to_inch, 0.22 * cm_to_inch))
    fig.subplots_adjust()
    plt.plot(speech)
    plt.axis("off")
    plt.ylim(-2500, 2500)
    plt.savefig(save_path / f"speech_waveform_{task}.png")
    plt.clf()
    plt.close()


def plot_eeg_emg_waveform(subject: str, date: str, session_name: str, save_path: Path):
    """plot raw eeg and emg

    Args:
        subject (str): subject name
        date (str): date of the experiment
        session_name (str): session name
        save_path (Path): path to save the figure
        task (str): task name
    """
    blue = tuple(np.array([47, 85, 151]) / 255)
    n_ch_eeg = 128
    n_ch_emg = 3

    before_preproc_paths = list(
        (
            Path(DATA_ROOT)
            / subject
            / date
            / "eeg_before_preproc"
            / f"{date}_{session_name}"
        ).glob("*.npy")
    )
    eeg = np.load(before_preproc_paths[2])
    eeg_z = zscore(eeg[:n_ch_eeg, :], axis=1)
    emg = np.vstack(
        (
            eeg[132, :] - eeg[133, :],
            eeg[134, :] - eeg[135, :],
            eeg[136, :] - eeg[137, :],
        )
    )
    emg = filter_data(emg, 256, 30, 127)
    emg_z = zscore(emg, axis=1)

    eeg_filtered, emg_filtered = preproc(eeg, wo_adapt_filt=False)
    eeg_filtered_z = zscore(eeg_filtered[:n_ch_eeg, :], axis=1)
    emg_filtered_z = zscore(emg, axis=1)

    for eeg_to_show, emg_to_show, name in zip(
        [eeg_z, eeg_filtered_z], [emg_z, emg_filtered_z], ["raw", "filtered"]
    ):
        fig, ax = plt.subplots(figsize=(3.0 * cm_to_inch, 2.5 * cm_to_inch))
        fig.subplots_adjust()
        for ch in range(n_ch_eeg):
            plt.plot(eeg_to_show[ch, :] - ch * 2, c="k")

        for i in range(n_ch_emg):
            plt.plot(emg_to_show[i, :] - (ch + 3) * 2 - i * 4, c=blue)
        plt.axis("off")
        plt.savefig(save_path / f"{name}_eeg_emg.png")
        plt.clf()
        plt.close()

    after_preproc_paths = list(
        (
            Path(DATA_ROOT)
            / subject
            / date
            / "eeg_after_preproc"
            / f"{date}_{session_name}"
        ).glob("*.pt")
    )
    eeg = torch.load(after_preproc_paths[2])
    eeg = eeg.to("cpu").detach().numpy()
    fig, ax = plt.subplots(figsize=(0.6 * cm_to_inch, 2.5 * cm_to_inch))
    fig.subplots_adjust()
    for ch in range(n_ch_eeg):
        plt.plot(eeg[0, 0, ch, :] - ch * 2, c="k")
    plt.axis("off")
    plt.savefig(save_path / f"filtered_avg_eeg.png")
    plt.clf()
    plt.close()


def show_likelihood_example():
    """show likelihood example
    Note that this is merely a schematic diagram and not a diagram derived from real data
    """
    green = tuple(np.array([0, 176, 80]) / 255)
    magenta = tuple(np.array([208, 0, 149]) / 255)
    orange = tuple(np.array([237, 125, 49]) / 255)
    violet = tuple(np.array([112, 48, 160]) / 255)
    yellow = tuple(np.array([255, 192, 0]) / 255)

    likelihood = [0.45, 0.23, 0.07, 0.17, 0.08]
    colors = [green, magenta, orange, violet, yellow]
    fig, ax = plt.subplots(figsize=(3.0 * cm_to_inch, 2.5 * cm_to_inch))
    fig.subplots_adjust()

    rect = ax.bar(np.arange(5), likelihood)
    for i, color in enumerate(colors):
        rect[i].set_color(color)
    plt.ylim([0, 1])
    plt.yticks([])
    plt.xticks([])
    ax.spines["right"].set_visible(True)
    ax.spines["top"].set_visible(True)
    plt.savefig("figures/fig1/likelihood_example.png")
    plt.clf()
    plt.close()


def main():
    save_path = Path("figures/fig1")
    save_path.mkdir(parents=True, exist_ok=True)
    dir_list = [
        "subject1-20230511",
        "subject1-20230529",
        "subject2-20230512",
        "subject2-20230516",
        "subject3-20230523",
        "subject3-20230524",
    ]
    for dir_ in dir_list:
        subject, date = dir_.split("-")
        cprint(f"{subject}, {date}", "cyan")
        check_file_existence(subject, date)

    data_tuples = [
        ("subject1", "20230529", "backup_calibrated_1", "overt"),
        ("subject1", "20230511", "backup_calibrated_1", "min-overt"),
        ("subject1", "20230529", "backup_calibrated_2", "covert"),
    ]
    for subject, date, session_name, task in data_tuples:
        plot_speech_waveform(subject, date, session_name, save_path, task)

    subject, date, session_name = "subject1", "20230511", "backup_calibrated_1"
    plot_eeg_emg_waveform(subject, date, session_name, save_path)

    show_likelihood_example()


if __name__ == "__main__":
    main()
