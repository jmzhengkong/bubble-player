from django.shortcuts import render
from django.http import HttpResponse
from django.contrib.staticfiles import finders
import numpy as np
import bubble_player


def index(request):
    return render(request, "bubble/index.html", {})


def map(request):
    current_map_id = int(np.random.random_integers(5))
    current_map = _load_map(current_map_id)

    request.session["map"] = current_map
    request.session["map_id"] = current_map_id
    request.session["score"] = 0
    context = {"buttons": _get_buttons(current_map), "score": 0}
    return render(request, "bubble/map.html", context)


def play(request):
    context = {
        "buttons": _get_buttons(request.session["map"]),
        "score": request.session["score"]}
    return render(request, "bubble/map.html", context)


def select(request, i, j):
    current_map = request.session["map"]
    previous_score = request.session["score"]
    next_map, score, selected = _update_map(current_map, i, j)
    request.session["map"] = next_map
    request.session["score"] += score
    request.session.modified = True
    context = {
        "buttons": _get_buttons(current_map, selected),
        "score": previous_score}
    return render(request, "bubble/select.html", context)


def machine(request):
    current_map = _load_map(request.session["map_id"])
    request.session["map"] = current_map
    request.session["score"] = 0
    request.session["step"] = 0
    request.session.modified = True
    context = {"buttons": _get_buttons(current_map), "score": 0}
    return render(request, "bubble/machine_map.html", context)


def machine_select(request):
    current_map = request.session["map"]
    previous_score = request.session["score"]
    step = request.session["step"]
    next_map, score, move = bubble_player.play(np.array(current_map), step)
    if next_map is None:
        context = {
            "buttons": _get_buttons(request.session["map"]),
            "score": request.session["score"]}
        return render(request, "bubble/map.html", context)
    selected = np.zeros([12, 11])
    for i, j in move:
        selected[i, j] = 1
    request.session["map"] = next_map.astype("int32").tolist()
    request.session["score"] += score
    request.session["step"] += 1
    request.session.modified = True
    context = {
        "buttons": _get_buttons(current_map, selected),
        "score": previous_score}
    return render(request, "bubble/machine_select.html", context)

def machine_play(request):
    context = {
        "buttons": _get_buttons(request.session["map"]),
        "score": request.session["score"]}
    return render(request, "bubble/machine_map.html", context)


def _get_buttons(current_map, selected=None):
    buttons = []
    for i in range(12):
        for j in range(11):
            if j == 0:
                buttons.append([])
            button = {}
            if current_map[i][j] == 0:
                button["class"] = "no-button"
            elif current_map[i][j] == 1:
                button["class"] = "yellow-button"
            elif current_map[i][j] == 2:
                button["class"] = "red-button"
            elif current_map[i][j] == 3:
                button["class"] = "green-button"
            elif current_map[i][j] == 4:
                button["class"] = "blue-button"
            else:
                button["class"] = "magenta-button"
            if selected is not None:
                if selected[i, j]:
                    button["background"] = "#DCDCDC"
            else:
                button["background"] = "white"
            buttons[-1].append(button)
    return buttons


def _update_map(current_map, i, j):
    current_map = np.array(current_map)
    if i > 0 and current_map[i - 1][j] == current_map[i][j]:
        stack = [(i, j)]
    elif j > 0 and current_map[i][j - 1] == current_map[i][j]:
        stack = [(i, j)]
    elif i < 11 and current_map[i + 1][j] == current_map[i][j]:
        stack = [(i, j)]
    elif j < 10 and current_map[i][j + 1] == current_map[i][j]:
        stack = [(i, j)]
    else:
        stack = []
    score = 0
    column_dict = {}
    selected = np.zeros([12, 11])
    while stack:
        x, y = stack.pop()
        if y not in column_dict:
            column_dict[y] = [i for i in range(12)]
        if x in column_dict[y]:
            column_dict[y].remove(x)
            score += 1
            selected[x, y] = 1
            if x > 0 and current_map[x - 1][y] == current_map[i][j]:
                stack.append((x - 1, y))
            if y > 0 and current_map[x][y - 1] == current_map[i][j]:
                stack.append((x, y - 1))
            if x < 11 and current_map[x + 1][y] == current_map[i][j]:
                stack.append((x + 1, y))
            if y < 10 and current_map[x][y + 1] == current_map[i][j]:
                stack.append((x, y + 1))
    for column, rows in column_dict.items():
        current_map[:, column] = np.concatenate(
            [np.zeros(12 - len(rows)), current_map[rows, column]])
    return current_map.astype("int32").tolist(), score * (score - 1), selected


def _load_map(current_map_id):
    current_map = []
    abs_path = finders.find("bubble/maps/{:d}.map".format(current_map_id))
    with open(abs_path, "r") as fp:
        for line in fp:
            current_map.append([int(e) for e in line.split(" ")[:11]])
    return current_map
