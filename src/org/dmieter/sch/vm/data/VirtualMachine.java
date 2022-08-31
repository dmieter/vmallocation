package org.dmieter.sch.vm.data;

import lombok.Getter;
import lombok.Setter;

@Setter
@Getter
public class VirtualMachine extends Entity {
    public Status status;
    public float performance;
    public int startupTime;

    public int timeStarted;
    public int timeDestroyed;

    public VirtualMachine(long id) {
        super(id);
    }

    public enum Status {operational, destroyed}

}
