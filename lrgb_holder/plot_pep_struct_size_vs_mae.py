import matplotlib.pyplot as plt

ticks = [500000, 220000, 100000, 50000, 25000, 12000, 7000, 1000]
labels = ['500k', '220k', '100k', '50k', '25k', '12K', '7k', '1K']


# SortMPNN
sort_num_params = [493858, 95970, 47328, 24942, 6718, 922]
sort_mae = [--, --, --, --, --]

# AdaptMPNN
adaptive_num_params = [478813, 98688, 49308, 24381, 6781, 982]
adaptive_mae = [--, --, --, --, --]

# GCN
gcn_num_params = [488061, 98451, 48851, 24255, 6821, 915]
gcn_mae = [--, --, --, --, --]


SMALL_SIZE = 24
MEDIUM_SIZE = 24
BIGGER_SIZE = 24

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

# Create the figure and axes
fig = plt.figure(figsize=(10, 6))
plt.xscale('log', base=2)
plt.xticks(ticks, labels)

# Plot the data for each method
plt.plot(sort_num_params, sort_mae, 'bo-', label='SortMPNN')
plt.plot(adaptive_num_params, adaptive_mae, 'rs--', label='AdaptMPNN')
plt.plot(gcn_num_params, gcn_mae, 'g^-.', label='GCN')

# Add labels and title
plt.xlabel('# Parameters' + r'[$log_2$]')
plt.ylabel('MAE')
plt.title('peptides-struct MAE vs. # parameters')
ax=plt.gca()
ax.invert_xaxis()

# Add legend and grid
plt.legend()
plt.grid(True)


# Show the plot
plt.show()