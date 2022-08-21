package org.dmieter.sch.vm.data;

import java.util.HashMap;
import java.util.Map;

public class Task {
    int computationsRequired;
    int taskDataVolume;
    Map<Task, Transfer> followingTasks = new HashMap<>();
}
