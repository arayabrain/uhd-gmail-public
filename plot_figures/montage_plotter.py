import tkinter as tk

import numpy as np

# クリックした座標を保存するリスト
clicked_coordinates = []


def on_click(event):
    x, y = event.x, event.y
    print(f"クリックした座標：x={x}, y={y}")
    # クリックした座標をリストに保存
    clicked_coordinates.append((x, y))

    # クリックした場所に赤い円を描画
    circle = canvas.create_oval(x - 5, y - 5, x + 5, y + 5, fill="red", outline="red")
    clicked_coordinates.append(circle)  # クリックした座標と円を関連付けて保存


def undo_click():
    if clicked_coordinates:
        # 最後のクリック情報を取得
        circle = clicked_coordinates.pop()
        item = clicked_coordinates.pop()
        if isinstance(item, tuple):
            # クリックした座標の場合、対応する円を削除
            x, y = item
            canvas.delete(item)  # クリックした座標を削除
            canvas.delete(circle)  # 関連する円を削除
            print(f"クリックを取り消し：x={x}, y={y}")


def exit_program():
    root.destroy()


# ウィンドウを作成
root = tk.Tk()
root.title("画像クリック座標取得")

# 画像を表示
image_path = "path/to/montage.png"  # 画像ファイルのパスを指定
img = tk.PhotoImage(file=image_path)
canvas = tk.Canvas(root, width=img.width(), height=img.height())
canvas.create_image(0, 0, anchor=tk.NW, image=img)
canvas.pack()

# クリックイベントを設定
canvas.bind("<Button-1>", on_click)

# 終了ボタンを追加
exit_button = tk.Button(root, text="終了", command=exit_program)
exit_button.pack()

# クリックを取り消すボタンを追加
undo_button = tk.Button(root, text="クリックを取り消す", command=undo_click)
undo_button.pack()

# ウィンドウを表示
root.mainloop()

# プログラムが終了したらクリックした座標を表示
coordinates = [x for x in clicked_coordinates if isinstance(x, tuple)]

np.save("coordinates.npy", coordinates)
print("クリックした回数:", len(coordinates))
print("クリックした座標リスト:", coordinates)
