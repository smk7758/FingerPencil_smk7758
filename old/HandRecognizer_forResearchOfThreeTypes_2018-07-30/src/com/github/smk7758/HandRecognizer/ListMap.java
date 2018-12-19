package com.github.smk7758.HandRecognizer;

import java.util.AbstractMap;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Map.Entry;
import java.util.Set;

public class ListMap<K, V> {
	private List<Entry<K, V>> list_map = new ArrayList<>();

	public int size() {
		return list_map.size();
	}

	public boolean add(K key, V value) {
		return list_map.add(createEntry(key, value));
	}

	private Entry<K, V> createEntry(K key, V value) {
		return new AbstractMap.SimpleEntry<K, V>(key, value);
	}

	public Entry<K, V> set(int index, K key, V value) {
		return list_map.set(index, createEntry(key, value));
	}

	public Entry<K, V> get(int index) {
		return list_map.get(index);
	}

	public K getKey(V value) {
		for (Entry<K, V> entry : list_map) {
			if (entry.getValue().equals(value)) return entry.getKey();
		}
		return null;
	}

	public V getValue(K key) {
		for (Entry<K, V> entry : list_map) {
			if (entry.getKey().equals(key)) return entry.getValue();
		}
		return null;
	}

	public List<Integer> getKeyIndexes(K key) {
		List<Integer> result_index = null;
		for (int index = 0; index < list_map.size(); index++) {
			Entry<K, V> entry = list_map.get(index);
			if (entry.getKey().equals(key)) {
				if (result_index == null) result_index = new ArrayList<>();
				result_index.add(index);
			}
		}
		return result_index;
	}

	public List<Integer> getValueIndexes(V value) {
		List<Integer> result_index = null;
		for (int index = 0; index < list_map.size(); index++) {
			Entry<K, V> entry = list_map.get(index);
			if (entry.getValue().equals(value)) {
				if (result_index == null) result_index = new ArrayList<>();
				result_index.add(index);
			}
		}
		return result_index;
	}

	public Entry<K, V> remove(int index) {
		return list_map.remove(index);
	}

	public void removeByKey(K key) {
		for (int index : getKeyIndexes(key)) {
			remove(index);
		}
	}

	public void removeByValue(V value) {
		for (int index : getValueIndexes(value)) {
			remove(index);
		}
	}

	public boolean containsKey(K key) {
		return getKeyIndexes(key) != null ? true : false;
	}

	public boolean containsValue(V value) {
		return getValueIndexes(value) != null ? true : false;
	}

	public Set<Entry<K, V>> entrySet() {
		return new HashSet<Entry<K, V>>(list_map);
	}
}
