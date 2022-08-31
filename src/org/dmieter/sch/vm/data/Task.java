package org.dmieter.sch.vm.data;

import java.util.HashMap;
import java.util.Map;

public class Task extends Entity {
    public Status status;
    public int computationsRequired;
    public int taskDataVolume;
    public Map<Task, Transfer> followingTasks = new HashMap<>();

    public Task(long id) {
        super(id);
    }

    public enum Status {queued, open, inprogress, completed}
}
