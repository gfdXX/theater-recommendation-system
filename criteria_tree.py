"""
Дерево критериев для фильтрации данных.
"""


class Node:
    """Класс для представления узла в дереве."""
    def __init__(self, value):
        self.value = value
        self.children = []

    def add_child(self, child):
        self.children.append(child)


class CriteriaTree:
    """Класс для создания и управления деревом критериев."""
    def __init__(self, df, columns):
        self.root = Node("Root")
        self.df = df
        self.columns = columns
        self.build_tree()

    def build_tree(self):
        self._build_recursive(self.root, self.df, 0)

    def _build_recursive(self, parent_node, current_df, col_index):
        if col_index >= len(self.columns):
            return

        column_name = self.columns[col_index]
        if current_df.empty:
            return

        unique_values = current_df[column_name].dropna().unique()

        for value in unique_values:
            new_node = Node(value)
            parent_node.add_child(new_node)

            next_df = current_df[current_df[column_name] == value]

            self._build_recursive(new_node, next_df, col_index + 1)

    def get_children_values(self, parent_node_value):
        def find_node(node, value):
            if node.value == value:
                return node
            for child in node.children:
                found_node = find_node(child, value)
                if found_node:
                    return found_node
            return None

        node = find_node(self.root, parent_node_value)
        if node:
            return [child.value for child in node.children]
        return []

    def print_tree(self):
        print("\n--- Полное дерево критериев ---")
        self._print_node(self.root, 0)

    def _print_node(self, node, level):
        if node.value != "Root":
            indent = "  " * level
            print(f"{indent}- {node.value}")

        for child in node.children:
            self._print_node(child, level + 1)

