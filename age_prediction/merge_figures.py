import sys
from PIL import Image

def main():
    output_path = sys.argv[1]
    image_paths = sys.argv[2:]
    left_col = [Image.open(path) for i, path in enumerate(image_paths) if i % 2 == 0]
    right_col = [Image.open(path) for i, path in enumerate(image_paths) if i % 2 != 0]

    grid_width = left_col[0].size[0] + right_col[0].size[0]
    grid_height = sum([img.size[1] for img in left_col])
    assert grid_height == sum([img.size[1] for img in right_col])

    grid_img = Image.new('RGB', (grid_width, grid_height))

    row_cursor = 0
    for i, (left_img, right_img) in enumerate(zip(left_col, right_col)):
        grid_img.paste(left_img, (0, row_cursor))
        grid_img.paste(right_img, (left_img.size[0], row_cursor))
        assert left_img.size[1] == right_img.size[1]
        row_cursor += left_img.size[1]
    
    grid_img.save(output_path)

if __name__ == '__main__':
    main()
