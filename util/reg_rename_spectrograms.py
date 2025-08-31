import os

# Set this to your folder
folder = r".\reg_spectrograms\\20_song"

# Get lists of .png and .pxp files, sorted by name
png_files = sorted([f for f in os.listdir(folder) if f.lower().endswith('.png')])
pxp_files = sorted([f for f in os.listdir(folder) if f.lower().endswith('.txt')])

# Check if counts match (for safety)
if len(png_files) != len(pxp_files):
    print("WARNING: Different number of .png and .txt files!")

# Rename files
for idx, (png, pxp) in enumerate(zip(png_files, pxp_files), start=1):
    num = idx
    new_png = f"image{num}.png"
    new_txt = f"image{num}.txt"
    os.rename(os.path.join(folder, png), os.path.join(folder, new_png))
    os.rename(os.path.join(folder, pxp), os.path.join(folder, new_txt))
    print(f"Renamed {png} -> {new_png}, {pxp} -> {new_txt}")

print("Renaming complete.")