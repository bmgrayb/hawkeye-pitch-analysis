import matplotlib as plt
import pandas as pd


def plot_pitch(pitch_df):
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(projection='3d')

    # get the indices of each joint type where each is closest to 0 (release time)
    # TODO: order by body part, not by z value
    release = pitch_df.groupby('hawkeye', as_index=False).agg({'time': lambda x: x.abs().min()})
    release_df = pd.merge(pitch_df, release, on=["hawkeye", "time"], how="inner").sort_values(by=['z'])
    ax.scatter(release_df.x, release_df.y, release_df.z, label='Release', s=50, edgecolors="black", alpha=1)

    # create connected plot to show shoulder -> elbow -> wrist in space
    plt.plot(release_df.x, release_df.y, release_df.z, linestyle='dashed', marker='s')

    # for each joint, plot the path in 3d
    for grp_name, grp_idx in pitch_df.groupby('hawkeye').groups.items():
        y = pitch_df.loc[grp_idx,'y']
        x = pitch_df.loc[grp_idx,'x']
        z = pitch_df.loc[grp_idx,'z']
        ax.scatter(x, y, z, label=grp_name, s=5, alpha=0.5)
        ax.set_xlabel('x-axis') 
        ax.set_ylabel('y-axis') 
        ax.set_zlabel('z-axis') 

    ax.legend(bbox_to_anchor=(1, 0.5), loc='center left', frameon=False)
    plt.show()